from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import AccountSummary, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.target import DailyTargetLedger


class DailyTargetLedgerTest(unittest.TestCase):
    def test_records_remaining_target_and_risk_budget_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="t1",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1000,
                        unrealized_pl_jpy=200.0,
                        take_profit=1.1020,
                        stop_loss=1.0990,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.1004, 1.1005, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now),
                },
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(
                start_balance_jpy=200_000,
                realized_pl_jpy=1000,
                daily_risk_budget_jpy=500,
                snapshot=snapshot,
            )

            self.assertEqual(summary.status, "PURSUE_TARGET")
            self.assertEqual(summary.target_jpy, 20_000)
            self.assertEqual(summary.minimum_target_jpy, 10_000)
            self.assertEqual(summary.progress_jpy, 1200)
            self.assertEqual(summary.remaining_minimum_jpy, 8_800)
            self.assertEqual(summary.remaining_target_jpy, 18_800)
            self.assertAlmostEqual(summary.remaining_risk_budget_jpy, 343.0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["minimum_return_pct"], 5.0)
            self.assertEqual(payload["minimum_progress_pct"], 12.0)
            self.assertEqual(payload["positions"][0]["remaining_risk_jpy"], 157.0)
            self.assertIn("Minimum daily floor", (root / "target.md").read_text())
            self.assertIn("Remaining target", (root / "target.md").read_text())

    def test_state_writes_canonical_campaign_day_and_as_of_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(
                start_balance_jpy=200_000,
                realized_pl_jpy=0,
                daily_risk_budget_jpy=500,
                now_utc=datetime(2026, 6, 24, 6, 0, tzinfo=timezone.utc),
            )

            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["campaign_day"], "2026-06-24")
            self.assertEqual(payload["campaign_day"], payload["campaign_day_jst"])
            self.assertEqual(payload["as_of_utc"], payload["generated_at_utc"])

    def test_accepts_absolute_target_profit_jpy_as_equity_derived_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(
                start_balance_jpy=200_000,
                target_profit_jpy=15_000,
                realized_pl_jpy=1500,
                daily_risk_budget_jpy=1000,
                target_trades_per_day=10,
            )

            self.assertEqual(summary.target_jpy, 15_000)
            self.assertEqual(summary.target_profit_jpy, 15_000)
            self.assertEqual(summary.minimum_target_jpy, 10_000)
            self.assertEqual(summary.remaining_target_jpy, 13_500)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["target_profit_jpy"], 15_000)
            self.assertEqual(payload["target_jpy"], 15_000)
            self.assertEqual(payload["target_return_pct"], 7.5)

    def test_usd_quote_position_risk_uses_snapshot_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="t1",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1000,
                        take_profit=1.1020,
                        stop_loss=1.0990,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"USD_JPY": Quote("USD_JPY", 199.99, 200.0, timestamp_utc=now)},
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(start_balance_jpy=200_000, daily_risk_budget_jpy=500, snapshot=snapshot)

            self.assertEqual(summary.status, "PURSUE_TARGET")
            self.assertEqual(summary.remaining_risk_budget_jpy, 300.0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["positions"][0]["remaining_risk_jpy"], 200.0)

    def test_unprotected_position_blocks_remaining_risk_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="t2",
                        pair="USD_JPY",
                        side=Side.SHORT,
                        units=1000,
                        entry_price=156.50,
                        unrealized_pl_jpy=-50.0,
                        take_profit=None,
                        stop_loss=None,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now)},
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(start_balance_jpy=100_000, daily_risk_budget_jpy=500, snapshot=snapshot)

            self.assertEqual(summary.status, "REPAIR_REQUIRED")
            self.assertEqual(summary.remaining_risk_budget_jpy, 0.0)
            self.assertEqual(summary.unprotected_positions, 1)
            payload = json.loads((root / "target.json").read_text())
            self.assertIn("TP", payload["positions"][0]["missing"])

    def test_sl_free_no_broker_tp_runner_does_not_zero_risk_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="471232",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=7000,
                        entry_price=1.16768,
                        unrealized_pl_jpy=-2149.5,
                        take_profit=None,
                        stop_loss=None,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"EUR_USD": Quote("EUR_USD", 1.16575, 1.16589, timestamp_utc=now)},
                account=AccountSummary(
                    nav_jpy=187_569.4,
                    balance_jpy=192_275.8,
                    unrealized_pl_jpy=-4_706.4,
                    fetched_at_utc=now,
                ),
            )

            prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
            try:
                summary = DailyTargetLedger(
                    state_path=root / "target.json",
                    report_path=root / "target.md",
                ).run(start_balance_jpy=192_578.9, daily_risk_budget_jpy=19_257.89, snapshot=snapshot)
            finally:
                if prior_sl is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
                if prior_tp is None:
                    os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
                else:
                    os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

            self.assertEqual(summary.status, "PURSUE_TARGET")
            self.assertEqual(summary.unprotected_positions, 0)
            self.assertEqual(summary.remaining_risk_budget_jpy, 19257.89)
            payload = json.loads((root / "target.json").read_text())
            self.assertTrue(payload["positions"][0]["protected"])
            self.assertIn("TP", payload["positions"][0]["missing"])
            self.assertNotIn("SL", payload["positions"][0]["missing"])

    def test_operator_manual_position_does_not_block_trader_risk_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="manual-470201",
                        pair="USD_JPY",
                        side=Side.LONG,
                        units=25000,
                        entry_price=155.962,
                        unrealized_pl_jpy=4650.0,
                        take_profit=None,
                        stop_loss=None,
                        owner=Owner.UNKNOWN,
                    ),
                ),
                quotes={"USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now)},
                account=AccountSummary(
                    nav_jpy=104_650.0,
                    balance_jpy=100_000.0,
                    unrealized_pl_jpy=4_650.0,
                    fetched_at_utc=now,
                ),
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(start_balance_jpy=100_000, daily_risk_budget_jpy=500, snapshot=snapshot)

            self.assertEqual(summary.status, "PURSUE_TARGET")
            self.assertEqual(summary.remaining_risk_budget_jpy, 500.0)
            self.assertEqual(summary.progress_jpy, 0.0)
            self.assertEqual(summary.remaining_target_jpy, 10_000.0)
            self.assertEqual(summary.unprotected_positions, 0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["unrealized_pl_jpy"], 0.0)
            self.assertEqual(payload["account_unrealized_pl_jpy"], 4650.0)
            self.assertEqual(payload["account_progress_jpy"], 4650.0)
            self.assertEqual(payload["current_equity_jpy"], 104650.0)
            self.assertEqual(payload["positions"][0]["owner"], "unknown")
            self.assertIn("TP", payload["positions"][0]["missing"])
            self.assertIn("SL", payload["positions"][0]["missing"])

    def test_updates_existing_target_without_repeating_start_balance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = DailyTargetLedger(state_path=root / "target.json", report_path=root / "target.md")
            ledger.run(start_balance_jpy=150_000, realized_pl_jpy=500)

            summary = ledger.run(realized_pl_jpy=1500)

            self.assertEqual(summary.target_jpy, 15_000)
            self.assertEqual(summary.progress_jpy, 1500)
            self.assertEqual(summary.remaining_target_jpy, 13_500)

    def test_same_day_snapshot_derives_realized_pl_from_account_balance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = DailyTargetLedger(state_path=root / "target.json", report_path=root / "target.md")
            now = datetime(2026, 5, 14, 1, 0, tzinfo=timezone.utc)

            ledger.run(
                start_balance_jpy=200_000,
                realized_pl_jpy=0,
                now_utc=now,
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="t4",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1000,
                        unrealized_pl_jpy=-200.0,
                        take_profit=1.1020,
                        stop_loss=1.0990,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now)},
                account=AccountSummary(
                    nav_jpy=198_300.0,
                    balance_jpy=198_500.0,
                    unrealized_pl_jpy=-200.0,
                    fetched_at_utc=now,
                ),
            )

            summary = ledger.run(snapshot=snapshot, now_utc=now)
            payload = json.loads((root / "target.json").read_text())

            self.assertEqual(summary.progress_jpy, -1700.0)
            self.assertEqual(summary.remaining_target_jpy, 21_700.0)
            self.assertEqual(payload["realized_pl_jpy"], -1500.0)
            self.assertEqual(payload["current_equity_jpy"], 198_300.0)

    def test_legacy_state_without_campaign_day_preserves_realized_pl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "target.json"
            state_path.write_text(
                json.dumps(
                    {
                        "start_balance_jpy": 100_000,
                        "target_return_pct": 10.0,
                        "realized_pl_jpy": 500.0,
                        "daily_risk_budget_jpy": 2_000,
                    }
                )
            )
            now = datetime(2026, 5, 14, 1, 0, tzinfo=timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                account=AccountSummary(
                    nav_jpy=200_000.0,
                    balance_jpy=200_000.0,
                    unrealized_pl_jpy=0.0,
                    fetched_at_utc=now,
                ),
            )

            summary = DailyTargetLedger(state_path=state_path, report_path=root / "target.md").run(
                snapshot=snapshot,
                now_utc=now,
            )
            payload = json.loads(state_path.read_text())

            self.assertEqual(summary.progress_jpy, 500.0)
            self.assertEqual(payload["realized_pl_jpy"], 500.0)

    def test_snapshotless_update_preserves_existing_open_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="t3",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1000,
                        unrealized_pl_jpy=-80.0,
                        take_profit=1.1020,
                        stop_loss=1.0990,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now)},
            )
            ledger = DailyTargetLedger(state_path=root / "target.json", report_path=root / "target.md")
            # Pass an explicit daily_risk_budget so the persistence assertion is unaffected
            # by the equity-derived default (which would otherwise be 200_000 * 2% = 4000).
            ledger.run(start_balance_jpy=200_000, daily_risk_budget_jpy=500, snapshot=snapshot)

            summary = ledger.run(start_balance_jpy=210_000)

            self.assertEqual(summary.target_jpy, 21_000)
            self.assertEqual(summary.progress_jpy, -80)
            self.assertAlmostEqual(summary.remaining_risk_budget_jpy, 343.0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["positions"][0]["trade_id"], "t3")

    def test_resets_target_on_new_jst_campaign_day_without_manual_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = DailyTargetLedger(state_path=root / "target.json", report_path=root / "target.md")

            first = ledger.run(
                start_balance_jpy=200_000,
                realized_pl_jpy=1_000,
                now_utc=datetime(2026, 5, 3, 0, 30, tzinfo=timezone.utc),
            )
            self.assertEqual(first.target_jpy, 20_000)

            second = ledger.run(now_utc=datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc))
            self.assertEqual(second.target_jpy, 20_100)
            self.assertEqual(second.progress_jpy, 0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["start_balance_jpy"], 201_000.0)
            self.assertEqual(payload["campaign_day_jst"], "2026-05-04")


    def test_auto_start_balance_from_snapshot_account_on_new_campaign_day(self) -> None:
        from quant_rabbit.models import AccountSummary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = DailyTargetLedger(state_path=root / "target.json", report_path=root / "target.md")

            ledger.run(
                start_balance_jpy=200_000,
                realized_pl_jpy=1_000,
                now_utc=datetime(2026, 5, 3, 0, 30, tzinfo=timezone.utc),
            )

            now = datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={"USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now)},
                account=AccountSummary(
                    nav_jpy=210_280.0,
                    balance_jpy=210_106.69,
                    unrealized_pl_jpy=173.31,
                    fetched_at_utc=now,
                ),
            )

            summary = ledger.run(snapshot=snapshot, now_utc=now)
            payload = json.loads((root / "target.json").read_text())

            self.assertAlmostEqual(payload["start_balance_jpy"], 210_106.69, places=2)
            self.assertEqual(payload["campaign_day_jst"], "2026-05-04")
            self.assertAlmostEqual(summary.target_jpy, 21_010.669, places=2)

    def test_auto_start_balance_first_run_with_no_previous_state(self) -> None:
        from quant_rabbit.models import AccountSummary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={"USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now)},
                account=AccountSummary(
                    nav_jpy=210_280.0,
                    balance_jpy=210_000.0,
                    unrealized_pl_jpy=280.0,
                    fetched_at_utc=now,
                ),
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(snapshot=snapshot)

            self.assertAlmostEqual(summary.target_jpy, 21_000.0, places=2)

    def test_per_trade_risk_splits_daily_budget_by_target_trade_pace(self) -> None:
        """Per-trade cap = daily_risk_budget / target_trades_per_day.

        Decoupling per-trade from per-day is what makes 'fire many small shots
        and let winners run' actually behave that way: a single losing trade
        spends only 1/N of the day's risk, so the trader can keep firing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
                target_trades_per_day=10,
            )

            self.assertEqual(summary.target_trades_per_day, 10)
            self.assertIsNone(summary.target_trades_per_day_basis_return_pct)
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 400.0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["target_trades_per_day"], 10)
            self.assertIsNone(payload["target_trades_per_day_basis_return_pct"])
            self.assertAlmostEqual(payload["per_trade_risk_budget_jpy"], 400.0)
            self.assertIn("Per-trade risk cap", (root / "target.md").read_text())

    def test_target_trades_per_day_falls_back_to_policy_default_with_documented_value(self) -> None:
        """When neither caller arg nor previous state sets the pace, RiskPolicy's
        documented default applies (currently 10) and is recorded in the snapshot.
        Per §3.5 the constant is documented, not silent."""
        from quant_rabbit.risk import RiskPolicy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(start_balance_jpy=200_000, daily_risk_budget_jpy=4000)

            self.assertEqual(summary.target_trades_per_day, RiskPolicy().target_trades_per_day)
            # The 1.0% min_per_trade_risk_pct floor (2026-06-11) lifts the
            # 4000/10=400 pace slice to 200_000 x 1.0% = 2_000 because this
            # pace is policy-derived, not operator-explicit CLI.
            self.assertAlmostEqual(
                summary.per_trade_risk_budget_jpy,
                200_000 * (RiskPolicy().min_per_trade_risk_pct / 100.0),
            )

    def test_target_trades_per_day_persists_across_runs(self) -> None:
        """Once an operator sets a pace, subsequent ledger runs keep it
        without requiring the flag again — mirrors how daily_risk_budget_jpy
        already persists."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            )
            ledger.run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
                target_trades_per_day=20,
            )

            summary = ledger.run(realized_pl_jpy=500)

            self.assertEqual(summary.target_trades_per_day, 20)
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 200.0)

    def test_cli_pace_persists_even_when_backtest_recommends_different(self) -> None:
        """Operator-explicit CLI pace persists across automation cycles even
        when ai_test_bot has a different recommendation. Regression seen
        2026-05-11: routine cycles invoked daily-target-state without args,
        which silently re-derived pace from ai_test_bot (capped to 30) and
        flipped per_trade from 1040 JPY (CLI choice) to 346 JPY, re-freezing
        attack-mode entries despite the per_trade unblock landing earlier.
        The operator's explicit `--target-trades-per-day` is treated as a
        deliberate choice; only an explicit override flips it.
        """
        from quant_rabbit.risk import RiskPolicy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            backtest.write_text(
                json.dumps(
                    {
                        "firepower": {
                            "avg_selected_trades_per_day": 12.0,
                            "required_trades_per_day_at_observed_expectancy": 25,
                        }
                    }
                )
            )
            ledger = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            )
            ledger.run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
                target_trades_per_day=10,
            )

            summary = ledger.run(realized_pl_jpy=0)
            payload = json.loads((root / "target.json").read_text())

            self.assertLessEqual(25, RiskPolicy().max_target_trades_per_day or 25)
            self.assertEqual(summary.target_trades_per_day, 10)
            self.assertEqual(summary.target_trades_per_day_source, "previous_cli")
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 4000.0 / 10, places=4)
            self.assertEqual(payload["target_trades_per_day_source"], "previous_cli")

            # Third cycle: previous source is now "previous_cli". The naive
            # `startswith("cli")` check would miss it and let ai_test_bot win.
            # Lock in that the carry-forward marker is also honored.
            summary3 = ledger.run(realized_pl_jpy=0)
            payload3 = json.loads((root / "target.json").read_text())
            self.assertEqual(summary3.target_trades_per_day, 10)
            self.assertEqual(summary3.target_trades_per_day_source, "previous_cli")
            self.assertEqual(payload3["target_trades_per_day_source"], "previous_cli")

    def test_cli_pace_expires_on_new_campaign_day_when_backtest_recommends_pace(self) -> None:
        """A previous-day CLI pace must not freeze the next campaign day.

        The same-day persistence test protects the operator from automation
        flipping sizing mid-campaign. On a new campaign day, however, carrying
        yesterday's CLI pace would bypass fresh ai-test-bot firepower evidence
        and recreate a fixed divisor.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            backtest.write_text(
                json.dumps(
                    {
                        "firepower": {
                            "avg_selected_trades_per_day": 12.0,
                            "required_trades_per_day_at_observed_expectancy": 25,
                        }
                    }
                )
            )
            ledger = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            )
            ledger.run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
                target_trades_per_day=10,
                now_utc=datetime(2026, 5, 11, 12, tzinfo=timezone.utc),
            )

            summary = ledger.run(
                realized_pl_jpy=0,
                now_utc=datetime(2026, 5, 12, 12, tzinfo=timezone.utc),
            )
            payload = json.loads((root / "target.json").read_text())

            self.assertEqual(summary.target_trades_per_day, 25)
            self.assertEqual(summary.target_trades_per_day_source, "ai_test_bot_required_trades_floored_by_min_per_trade_pct")
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 2000.0, places=4)  # 1.0% equity floor > 4000/25
            self.assertEqual(payload["target_trades_per_day_source"], "ai_test_bot_required_trades_floored_by_min_per_trade_pct")

    def test_target_band_pace_targets_next_attainable_band_before_10pct_firepower(self) -> None:
        """5-10% adjustment must flow into execution pace, not only reports.

        When the selected policy has reached the 5% floor but not 10%, the
        next execution loop should tune against the next measurable band
        (6% here) while preserving the 5% floor. Falling back to the 10%
        firepower number over-thins per-trade risk before the opportunity
        universe can actually support that stretch target.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            backtest.write_text(
                json.dumps(
                    {
                        "target_return_pct": 10.0,
                        "firepower": {
                            "avg_selected_trades_per_day": 6.0,
                            "required_trades_per_day_at_observed_expectancy": 80,
                        },
                        "target_band": {
                            "floor_return_pct": 5.0,
                            "stretch_return_pct": 10.0,
                            "selected_attainable_return_pct": 5.0,
                            "bands": [
                                {
                                    "return_pct": 5.0,
                                    "required_trades_per_day_at_observed_expectancy": 20,
                                },
                                {
                                    "return_pct": 6.0,
                                    "required_trades_per_day_at_observed_expectancy": 23,
                                },
                                {
                                    "return_pct": 10.0,
                                    "required_trades_per_day_at_observed_expectancy": 80,
                                },
                            ],
                        },
                    }
                )
            )
            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            ).run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
            )
            payload = json.loads((root / "target.json").read_text())
            report = (root / "target.md").read_text()

            self.assertEqual(summary.target_trades_per_day, 23)
            self.assertEqual(
                summary.target_trades_per_day_source,
                "ai_test_bot_target_band_6pct_required_trades_floored_by_min_per_trade_pct",
            )
            self.assertEqual(summary.target_trades_per_day_basis_return_pct, 6.0)
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 2000.0, places=4)  # 1.0% equity floor > 4000/23
            self.assertEqual(payload["target_trades_per_day_basis_return_pct"], 6.0)
            self.assertIn("Target trade pace basis: `6.0%`", report)

    def test_target_sizing_near_miss_can_reduce_pace_for_floor_attempt(self) -> None:
        """A verified 5% near miss should size from target_sizing, not 10% firepower.

        This is still bounded by the same daily risk budget: lowering the pace
        from the capped required-trades number raises the per-trade slice only
        enough to test the measured floor near miss.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            backtest.write_text(
                json.dumps(
                    {
                        "target_return_pct": 10.0,
                        "daily_risk_budget_jpy": 4000.0,
                        "firepower": {
                            "avg_selected_trades_per_day": 6.0,
                            "required_trades_per_day_at_observed_expectancy": 80,
                        },
                        "target_band": {
                            "floor_return_pct": 5.0,
                            "stretch_return_pct": 10.0,
                            "selected_attainable_return_pct": None,
                            "bands": [
                                {
                                    "return_pct": 5.0,
                                    "required_trades_per_day_at_observed_expectancy": 52,
                                },
                                {
                                    "return_pct": 10.0,
                                    "required_trades_per_day_at_observed_expectancy": 80,
                                },
                            ],
                        },
                        "mechanism_ablation": {
                            "target_sizing": {
                                "status": "FLOOR_NEAR_MISS_SIZE_TEST",
                                "bands": [
                                    {
                                        "return_pct": 5.0,
                                        "required_size_multiplier": 1.0314,
                                        "scaled_loss_cap_jpy": 140.0,
                                        "status": "NEAR_MISS_SIZE_TEST",
                                    }
                                ],
                            }
                        },
                    }
                )
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            ).run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
            )
            payload = json.loads((root / "target.json").read_text())

            self.assertEqual(summary.target_trades_per_day, 28)
            self.assertEqual(
                summary.target_trades_per_day_source,
                "ai_test_bot_target_sizing_5pct_near_miss_floored_by_min_per_trade_pct",
            )
            self.assertEqual(summary.target_trades_per_day_basis_return_pct, 5.0)
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 2000.0, places=4)  # 1.0% equity floor > 4000/28
            self.assertEqual(payload["target_trades_per_day_basis_return_pct"], 5.0)

    def test_target_sizing_moderate_floor_candidate_updates_live_pace(self) -> None:
        """A measured moderate 5% floor size-up must reach execution sizing.

        Regression: ai-test-bot reported a 5% floor candidate at about 1.20x
        size, but DailyTargetLedger only consumed <=1.10x near misses. That
        left the campaign stuck on the 5% firepower pace cap, so the report
        identified the sizing fix while live intents kept using the thinner
        per-trade cap.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            backtest.write_text(
                json.dumps(
                    {
                        "target_return_pct": 10.0,
                        "daily_risk_budget_jpy": 22278.1,
                        "firepower": {
                            "avg_selected_trades_per_day": 4.8,
                            "required_trades_per_day_at_observed_expectancy": 125,
                        },
                        "target_band": {
                            "floor_return_pct": 5.0,
                            "stretch_return_pct": 10.0,
                            "selected_attainable_return_pct": None,
                            "bands": [
                                {
                                    "return_pct": 5.0,
                                    "required_trades_per_day_at_observed_expectancy": 63,
                                },
                                {
                                    "return_pct": 10.0,
                                    "required_trades_per_day_at_observed_expectancy": 125,
                                },
                            ],
                        },
                        "mechanism_ablation": {
                            "target_sizing": {
                                "status": "FLOOR_MODERATE_SIZE_UP_REQUIRED",
                                "bands": [
                                    {
                                        "return_pct": 5.0,
                                        "required_size_multiplier": 1.1972,
                                        "scaled_loss_cap_jpy": 889.0447,
                                        "scaled_target_hit_days": 1,
                                        "scaled_max_drawdown_jpy": 2272.4467,
                                        "scaled_worst_day_jpy": -1198.8139,
                                        "status": "MODERATE_SIZE_UP_REQUIRED",
                                    }
                                ],
                            }
                        },
                    }
                )
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            ).run(
                start_balance_jpy=222_781,
                daily_risk_budget_jpy=22278.1,
            )
            payload = json.loads((root / "target.json").read_text())

            self.assertEqual(summary.target_trades_per_day, 25)
            self.assertEqual(
                summary.target_trades_per_day_source,
                "ai_test_bot_target_sizing_5pct_moderate_floored_by_min_per_trade_pct",
            )
            self.assertEqual(summary.target_trades_per_day_basis_return_pct, 5.0)
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 2227.81, places=4)  # 1.0% equity floor > 22278.1/25
            self.assertEqual(payload["target_trades_per_day_basis_return_pct"], 5.0)

    def test_target_sizing_moderate_rejects_daily_risk_breach(self) -> None:
        """Moderate sizing is not accepted when the scaled risk envelope breaks."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            backtest.write_text(
                json.dumps(
                    {
                        "target_return_pct": 10.0,
                        "daily_risk_budget_jpy": 4000.0,
                        "firepower": {
                            "avg_selected_trades_per_day": 4.8,
                            "required_trades_per_day_at_observed_expectancy": 80,
                        },
                        "target_band": {
                            "floor_return_pct": 5.0,
                            "stretch_return_pct": 10.0,
                            "selected_attainable_return_pct": None,
                            "bands": [
                                {
                                    "return_pct": 5.0,
                                    "required_trades_per_day_at_observed_expectancy": 52,
                                }
                            ],
                        },
                        "mechanism_ablation": {
                            "target_sizing": {
                                "status": "FLOOR_MODERATE_SIZE_UP_REQUIRED",
                                "bands": [
                                    {
                                        "return_pct": 5.0,
                                        "required_size_multiplier": 1.2,
                                        "scaled_loss_cap_jpy": 900.0,
                                        "scaled_target_hit_days": 1,
                                        "scaled_max_drawdown_jpy": 4500.0,
                                        "scaled_worst_day_jpy": -1200.0,
                                        "status": "MODERATE_SIZE_UP_REQUIRED",
                                    }
                                ],
                            }
                        },
                    }
                )
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            ).run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
            )

            self.assertEqual(summary.target_trades_per_day, 30)
            self.assertEqual(
                summary.target_trades_per_day_source,
                "ai_test_bot_target_band_5pct_required_trades_capped_floored_by_min_per_trade_pct",
            )

    def test_previous_cli_pace_yields_to_fresh_target_band_backtest(self) -> None:
        """A carried CLI pace must not block a freshly regenerated band pace.

        The first no-override cycle preserves an operator-set CLI pace. After
        the automation refreshes ai-test-bot in the same campaign, the carried
        `previous_cli` marker can yield to the fresher 5-10% band evidence.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            state = root / "target.json"
            ledger = DailyTargetLedger(
                state_path=state,
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            )
            ledger.run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
                target_trades_per_day=10,
            )
            previous = json.loads(state.read_text())
            previous["target_trades_per_day_source"] = "previous_cli"
            previous["generated_at_utc"] = "2026-05-11T10:00:00+00:00"
            state.write_text(json.dumps(previous))
            backtest.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-11T10:01:00+00:00",
                        "target_return_pct": 10.0,
                        "firepower": {
                            "avg_selected_trades_per_day": 6.0,
                            "required_trades_per_day_at_observed_expectancy": 80,
                        },
                        "target_band": {
                            "floor_return_pct": 5.0,
                            "stretch_return_pct": 10.0,
                            "selected_attainable_return_pct": 5.0,
                            "bands": [
                                {
                                    "return_pct": 5.0,
                                    "required_trades_per_day_at_observed_expectancy": 20,
                                },
                                {
                                    "return_pct": 6.0,
                                    "required_trades_per_day_at_observed_expectancy": 23,
                                },
                            ],
                        },
                    }
                )
            )

            summary = ledger.run(
                realized_pl_jpy=0,
                now_utc=datetime(2026, 5, 11, 12, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.target_trades_per_day, 23)
            self.assertEqual(
                summary.target_trades_per_day_source,
                "ai_test_bot_target_band_6pct_required_trades_floored_by_min_per_trade_pct",
            )
            self.assertEqual(summary.target_trades_per_day_basis_return_pct, 6.0)

    def test_backtest_required_pace_is_capped_to_policy_ceiling(self) -> None:
        """An absurd ai-test-bot pace must not silently shrink per-trade sizing.

        Regression: ai-test-bot.firepower returned 229 required trades/day when
        observed expectancy was too thin to hit the 10% target. That number
        flowed through unfiltered to per_trade_risk_budget_jpy ≈ 18 JPY, which
        sized live EUR_USD orders at ~200 units (≈0.02 standard lot), making
        execution operationally meaningless. The cap keeps sizing tradable;
        the strategy/expectancy gap is still surfaced by ai_test_bot.firepower.
        """
        from quant_rabbit.risk import RiskPolicy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtest = root / "ai_test_bot.json"
            backtest.write_text(
                json.dumps(
                    {
                        "firepower": {
                            "avg_selected_trades_per_day": 6.1892,
                            "required_trades_per_day_at_observed_expectancy": 229,
                        }
                    }
                )
            )
            ledger = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                pace_backtest_path=backtest,
            )

            summary = ledger.run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
            )
            payload = json.loads((root / "target.json").read_text())

            cap = RiskPolicy().max_target_trades_per_day
            self.assertIsNotNone(cap)
            self.assertEqual(summary.target_trades_per_day, cap)
            self.assertEqual(
                summary.target_trades_per_day_source,
                "ai_test_bot_required_trades_capped_floored_by_min_per_trade_pct",
            )
            self.assertAlmostEqual(
                # floored: 200_000 x min_per_trade_risk_pct (1.0%) > 4000/cap
                summary.per_trade_risk_budget_jpy,
                2000.0,
                places=4,
            )
            self.assertEqual(
                payload["target_trades_per_day_source"],
                "ai_test_bot_required_trades_capped_floored_by_min_per_trade_pct",
            )

    def test_explicit_cli_pace_is_not_capped(self) -> None:
        """Operator-set pace via --target-trades-per-day must pass through.

        The cap exists to prevent silent backtest-driven sabotage. An explicit
        operator override is the documented way to declare a higher pace
        (e.g. for chance-time S-size attacks); the cap must not clip it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            )

            summary = ledger.run(
                start_balance_jpy=200_000,
                daily_risk_budget_jpy=4000,
                target_trades_per_day=200,
            )

            self.assertEqual(summary.target_trades_per_day, 200)
            self.assertEqual(summary.target_trades_per_day_source, "cli")

    def test_explicit_start_balance_overrides_snapshot_account(self) -> None:
        from quant_rabbit.models import AccountSummary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={"USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now)},
                account=AccountSummary(
                    nav_jpy=210_280.0,
                    balance_jpy=210_000.0,
                    fetched_at_utc=now,
                ),
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            ).run(start_balance_jpy=100_000, snapshot=snapshot)

            self.assertAlmostEqual(summary.target_jpy, 10_000.0, places=2)

    def test_first_run_reconstructs_current_day_realized_from_execution_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "execution_ledger.db"
            with sqlite3.connect(ledger_path) as conn:
                conn.execute(
                    "CREATE TABLE execution_events (ts_utc TEXT, event_type TEXT, realized_pl_jpy REAL)"
                )
                conn.execute(
                    "CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT, updated_at_utc TEXT)"
                )
                conn.execute(
                    "INSERT INTO sync_state VALUES ('last_oanda_transaction_id', '1', '2026-05-15T00:45:00+00:00')"
                )
                conn.executemany(
                    "INSERT INTO execution_events (ts_utc, event_type, realized_pl_jpy) VALUES (?, ?, ?)",
                    [
                        ("2026-05-15T00:30:00+00:00", "TRADE_CLOSED", -5641.44),
                        ("2026-05-14T23:30:00+00:00", "TRADE_CLOSED", 1000.0),
                    ],
                )
            now = datetime(2026, 5, 15, 1, 0, tzinfo=timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="t1",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1000,
                        unrealized_pl_jpy=-300.0,
                        take_profit=1.1020,
                        stop_loss=1.0990,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.1004, 1.1005, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 149.99, 150.0, timestamp_utc=now),
                },
                account=AccountSummary(
                    nav_jpy=193_300.0,
                    balance_jpy=193_600.0,
                    unrealized_pl_jpy=-300.0,
                    fetched_at_utc=now,
                ),
            )

            summary = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
                execution_ledger_path=ledger_path,
            ).run(snapshot=snapshot, now_utc=now)

            self.assertAlmostEqual(summary.progress_jpy, -5941.44)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["campaign_day_jst"], "2026-05-15")
            self.assertAlmostEqual(payload["start_balance_jpy"], 199241.44)
            self.assertAlmostEqual(payload["realized_pl_jpy"], -5641.44)
            self.assertAlmostEqual(payload["progress_jpy"], -5941.44)

    def test_same_day_poisoned_start_balance_self_heals_from_ledger_truth(self) -> None:
        """§3.5 stale-persistence repair (2026-06-08 incident regression).

        A persisted start_balance from weeks earlier (222,781) against a
        current broker balance of 184,962 must not report -37,818 JPY of
        same-day "realized" loss. With execution-ledger truth available the
        ledger figure wins and the start balance re-derives from
        `balance - realized_today`.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "execution_ledger.db"
            with sqlite3.connect(ledger_path) as conn:
                conn.execute(
                    "CREATE TABLE execution_events (ts_utc TEXT, event_type TEXT, realized_pl_jpy REAL)"
                )
                conn.execute(
                    "CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT, updated_at_utc TEXT)"
                )
                conn.execute(
                    "INSERT INTO sync_state VALUES ('last_oanda_transaction_id', '472132', '2026-06-08T17:00:00+00:00')"
                )
                conn.executemany(
                    "INSERT INTO execution_events (ts_utc, event_type, realized_pl_jpy) VALUES (?, ?, ?)",
                    [
                        ("2026-06-08T01:26:00+00:00", "TRADE_CLOSED", 433.68),
                        ("2026-06-08T05:03:00+00:00", "TRADE_CLOSED", -172.86),
                        ("2026-06-08T16:38:00+00:00", "TRADE_CLOSED", -169.16),
                        # Prior-day rows must not leak into the campaign day.
                        ("2026-06-05T12:41:00+00:00", "TRADE_CLOSED", -2981.90),
                    ],
                )
            state_path = root / "target.json"
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_day_jst": "2026-06-08",
                        "start_balance_jpy": 222_781.0,
                        "current_equity_jpy": 184_962.9332,
                        "target_return_pct": 10.0,
                        "realized_pl_jpy": -37_818.0668,
                        "daily_risk_pct": 10.0,
                    }
                )
            )
            now = datetime(2026, 6, 8, 17, 18, tzinfo=timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                account=AccountSummary(
                    nav_jpy=184_962.9332,
                    balance_jpy=184_962.9332,
                    unrealized_pl_jpy=0.0,
                    fetched_at_utc=now,
                ),
            )

            summary = DailyTargetLedger(
                state_path=state_path,
                report_path=root / "target.md",
                execution_ledger_path=ledger_path,
            ).run(snapshot=snapshot, now_utc=now)
            payload = json.loads(state_path.read_text())

            expected_realized = 433.68 - 172.86 - 169.16
            self.assertEqual(payload["campaign_day_jst"], "2026-06-08")
            self.assertAlmostEqual(payload["realized_pl_jpy"], expected_realized, places=2)
            self.assertAlmostEqual(
                payload["start_balance_jpy"], 184_962.9332 - expected_realized, places=2
            )
            # Daily risk budget re-derives from the healed start balance.
            self.assertAlmostEqual(
                payload["daily_risk_budget_jpy"],
                round((184_962.9332 - expected_realized) * 0.10, 4),
                places=2,
            )
            self.assertAlmostEqual(summary.progress_jpy, expected_realized, places=2)

    def test_same_day_without_ledger_keeps_balance_diff_realized(self) -> None:
        """Without an execution ledger the snapshot balance-diff fallback stays."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "target.json"
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_day_jst": "2026-06-08",
                        "start_balance_jpy": 200_000.0,
                        "current_equity_jpy": 200_000.0,
                        "target_return_pct": 10.0,
                        "realized_pl_jpy": 0.0,
                    }
                )
            )
            now = datetime(2026, 6, 8, 17, 18, tzinfo=timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                account=AccountSummary(
                    nav_jpy=198_500.0,
                    balance_jpy=198_500.0,
                    unrealized_pl_jpy=0.0,
                    fetched_at_utc=now,
                ),
            )

            summary = DailyTargetLedger(
                state_path=state_path,
                report_path=root / "target.md",
            ).run(snapshot=snapshot, now_utc=now)
            payload = json.loads(state_path.read_text())

            # No audited realized figure: start balance must NOT re-anchor.
            self.assertAlmostEqual(payload["start_balance_jpy"], 200_000.0)
            self.assertAlmostEqual(payload["realized_pl_jpy"], -1_500.0)
            self.assertAlmostEqual(summary.progress_jpy, -1_500.0)


if __name__ == "__main__":
    unittest.main()

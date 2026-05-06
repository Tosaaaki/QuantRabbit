from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
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
            self.assertEqual(summary.progress_jpy, 1200)
            self.assertEqual(summary.remaining_target_jpy, 18_800)
            self.assertAlmostEqual(summary.remaining_risk_budget_jpy, 343.0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["positions"][0]["remaining_risk_jpy"], 157.0)
            self.assertIn("Remaining target", (root / "target.md").read_text())

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
            self.assertAlmostEqual(summary.per_trade_risk_budget_jpy, 400.0)
            payload = json.loads((root / "target.json").read_text())
            self.assertEqual(payload["target_trades_per_day"], 10)
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
            self.assertAlmostEqual(
                summary.per_trade_risk_budget_jpy,
                4000.0 / RiskPolicy().target_trades_per_day,
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


if __name__ == "__main__":
    unittest.main()

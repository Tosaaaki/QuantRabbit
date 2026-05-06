"""Tests guarding the 'no hardcoded JPY fallback' invariant on the risk path.

These tests pin down the contract that:
  1. RiskPolicy.max_loss_jpy may be None (caller must inject the cap).
  2. validate() emits LOSS_CAP_MISSING (rather than silently using a literal)
     when no per-trade cap is reachable from policy or intent metadata.
  3. intent.metadata['max_loss_jpy'] overrides policy.max_loss_jpy so the
     intent generator can pass an equity-derived cap per lane.
  4. resolve_max_loss_jpy() raises (no silent literal fallback) when neither
     explicit value nor default is provided.
  5. DailyTargetLedger derives daily_risk_budget_jpy from start_balance and
     RiskPolicy.daily_risk_pct when the caller does not pass an explicit value.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import (
    AccountSummary,
    BrokerSnapshot,
    MarketContext,
    OrderIntent,
    OrderType,
    Owner,
    Quote,
    Side,
    TradeMethod,
)
from quant_rabbit.risk import RiskEngine, RiskPolicy, resolve_max_loss_jpy
from quant_rabbit.target import DailyTargetLedger


def _intent(*, metadata: dict | None = None) -> OrderIntent:
    return OrderIntent(
        pair="EUR_USD",
        side=Side.LONG,
        order_type=OrderType.STOP_ENTRY,
        units=10_000,
        entry=1.1735,
        tp=1.1755,
        sl=1.1715,
        thesis="test lane: equity-derived cap",
        owner=Owner.TRADER,
        market_context=MarketContext(
            regime="TREND_UP M5/M15 aligned",
            narrative="risk-on flow into EU session",
            chart_story="reclaim of prior swing high after retest",
            method=TradeMethod.TREND_CONTINUATION,
            invalidation="break back below SL",
            event_risk="",
            session="EU",
        ),
        metadata=metadata,
    )


def _snapshot() -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        quotes={
            "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
            "USD_JPY": Quote("USD_JPY", 156.99, 157.0, timestamp_utc=now),
        },
        account=AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            margin_used_jpy=0.0,
            margin_available_jpy=200_000.0,
            fetched_at_utc=now,
        ),
    )


class RiskNoHardcodeTest(unittest.TestCase):
    def test_risk_policy_accepts_none_max_loss_jpy(self) -> None:
        policy = RiskPolicy(max_loss_jpy=None)
        self.assertIsNone(policy.max_loss_jpy)

    def test_validate_emits_loss_cap_missing_when_no_cap_reachable(self) -> None:
        engine = RiskEngine(policy=RiskPolicy(max_loss_jpy=None))
        decision = engine.validate(_intent(), _snapshot())
        codes = {issue.code for issue in decision.issues}
        self.assertIn("LOSS_CAP_MISSING", codes)
        self.assertNotIn("LOSS_CAP_EXCEEDED", codes)

    def test_intent_metadata_cap_overrides_policy_cap(self) -> None:
        # Policy cap = 500 JPY would exceed the lane's worst-case risk;
        # intent metadata cap = 5_000 JPY is more permissive and must win.
        engine = RiskEngine(policy=RiskPolicy(max_loss_jpy=500.0))
        intent = _intent(metadata={"max_loss_jpy": 5_000.0})
        decision = engine.validate(intent, _snapshot())
        codes = {issue.code for issue in decision.issues}
        # With the equity-derived cap of 5000 JPY, the lane risk fits and the
        # validator does not raise LOSS_CAP_EXCEEDED.
        self.assertNotIn("LOSS_CAP_EXCEEDED", codes)
        self.assertNotIn("LOSS_CAP_MISSING", codes)

    def test_resolve_max_loss_jpy_raises_without_default(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_max_loss_jpy(
                max_loss_jpy=None,
                max_loss_pct=None,
                equity_jpy=None,
                default_max_loss_jpy=None,
                label="unit test",
            )
        self.assertIn("No JPY literal fallback", str(ctx.exception))

    def test_daily_target_ledger_derives_budget_from_equity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = DailyTargetLedger(
                state_path=root / "target.json",
                report_path=root / "target.md",
            )
            # No explicit daily_risk_budget_jpy → must derive from equity.
            summary = ledger.run(start_balance_jpy=200_000)
            # 200_000 * (RiskPolicy.daily_risk_pct=2.0%) = 4000 JPY
            self.assertAlmostEqual(summary.remaining_risk_budget_jpy, 4000.0)


if __name__ == "__main__":
    unittest.main()


class GeometryAtrTest(unittest.TestCase):
    """Geometry must be ATR-derived; missing pair_charts must surface as a BLOCK."""

    def test_geometry_uses_atr_when_provided(self) -> None:
        from datetime import datetime, timezone
        from quant_rabbit.models import OrderType, Quote, Side
        from quant_rabbit.strategy.intent_generator import (
            GEOMETRY_ATR_MULT,
            GEOMETRY_SPREAD_FLOOR_MULT,
            _geometry,
        )

        now = datetime.now(timezone.utc)
        quote = Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now)
        # Spread = 0.8 pip, so spread floor = 0.8 * 6 = 4.8 pips.
        # Pass an ATR larger than the floor so ATR dominates the stop choice.
        atr_pips = 12.0
        entry, tp, sl = _geometry(
            "EUR_USD",
            Side.LONG,
            OrderType.STOP_ENTRY,
            quote,
            reward_risk=2.0,
            atr_pips=atr_pips,
        )
        # SL distance from entry should equal atr_pips * GEOMETRY_ATR_MULT.
        sl_pips = (entry - sl) * 10000
        expected_stop_pips = max(atr_pips * GEOMETRY_ATR_MULT, 0.8 * GEOMETRY_SPREAD_FLOOR_MULT)
        self.assertAlmostEqual(sl_pips, expected_stop_pips, places=2)

    def test_geometry_falls_to_spread_floor_when_atr_smaller(self) -> None:
        from datetime import datetime, timezone
        from quant_rabbit.models import OrderType, Quote, Side
        from quant_rabbit.strategy.intent_generator import (
            GEOMETRY_SPREAD_FLOOR_MULT,
            _geometry,
        )

        now = datetime.now(timezone.utc)
        quote = Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now)
        # Tiny ATR — spread floor (0.8 * 6 = 4.8 pips) must dominate.
        entry, tp, sl = _geometry(
            "EUR_USD",
            Side.LONG,
            OrderType.STOP_ENTRY,
            quote,
            reward_risk=2.0,
            atr_pips=1.0,
        )
        sl_pips = (entry - sl) * 10000
        self.assertAlmostEqual(sl_pips, 0.8 * GEOMETRY_SPREAD_FLOOR_MULT, places=2)

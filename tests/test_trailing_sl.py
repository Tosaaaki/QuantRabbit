"""Coverage for 2026-05-13 broker-side trailing SL update logic.

The 2026-05-12T15:33 mass-close incident drove the operator demand:
new entries must carry a broker-side SL, and that SL must trail
adverse M15/M30/H1 structure prints so a panic-close trader cannot
override the broker protection. Tests pin three invariants:

1. Positions WITHOUT a broker SL (SL-free mode) are NEVER touched —
   this is the "existing-position absolute protection" guarantee.
2. SL is only ever TIGHTENED, never widened (mirrors AGENT_CONTRACT
   §10: "Existing SL cannot be widened").
3. Only thesis-reverse BOS/CHOCH events on M15/M30/H1 trigger
   updates; same-direction events do not move SL.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from quant_rabbit.models import (
    AccountSummary,
    BrokerPosition,
    BrokerSnapshot,
    Owner,
    Quote,
    Side,
)
from quant_rabbit.strategy.trailing_sl import (
    TRAILING_SL_SPREAD_BUFFER_MULT,
    TRAILING_TIMEFRAMES,
    _compute_new_sl,
    _parse_struct_events,
    apply_trailing_sls,
)


def _snapshot(*, positions=(), quotes=None) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=tuple(positions),
        orders=tuple(),
        quotes=quotes or {
            "EUR_USD": Quote("EUR_USD", bid=1.17300, ask=1.17310, timestamp_utc=now),
            "AUD_JPY": Quote("AUD_JPY", bid=113.700, ask=113.716, timestamp_utc=now),
        },
        account=AccountSummary(
            balance_jpy=200_000.0,
            nav_jpy=200_000.0,
            margin_used_jpy=50_000.0,
            margin_available_jpy=150_000.0,
            fetched_at_utc=now,
        ),
    )


def _pair_charts(**chart_stories) -> dict:
    return {"charts": [{"pair": pair, "chart_story": story} for pair, story in chart_stories.items()]}


def _long_position(sl: float | None = 1.17000, trade_id: str = "9000001") -> BrokerPosition:
    return BrokerPosition(
        trade_id=trade_id,
        pair="EUR_USD",
        side=Side.LONG,
        units=5000,
        entry_price=1.17500,
        unrealized_pl_jpy=-50.0,
        take_profit=1.18000,
        stop_loss=sl,
        owner=Owner.TRADER,
    )


def _short_position(sl: float | None = 114.500, trade_id: str = "9000002") -> BrokerPosition:
    return BrokerPosition(
        trade_id=trade_id,
        pair="AUD_JPY",
        side=Side.SHORT,
        units=5000,
        entry_price=113.700,
        unrealized_pl_jpy=-30.0,
        take_profit=113.000,
        stop_loss=sl,
        owner=Owner.TRADER,
    )


class StructParserTest(unittest.TestCase):
    def test_parses_m15_h1_only(self) -> None:
        events = _parse_struct_events(
            "EUR_USD; M15(struct=BOS_DOWN@1.17200); H1(struct=CHOCH_UP@1.18000); "
            "H4(struct=BOS_UP@1.20000); M5(struct=BOS_DOWN@1.17300)"
        )
        self.assertEqual(events.get("M15"), ("BOS", "DOWN", 1.17200))
        self.assertEqual(events.get("H1"), ("CHOCH", "UP", 1.18000))
        # All TFs parsed; consumer filters by TRAILING_TIMEFRAMES.
        self.assertIn("M5", events)
        self.assertIn("H4", events)

    def test_returns_empty_for_missing_string(self) -> None:
        self.assertEqual(_parse_struct_events(""), {})
        self.assertEqual(_parse_struct_events("no struct here"), {})


class ComputeNewSlTest(unittest.TestCase):
    def test_long_tightens_sl_below_bos_minus_buffer(self) -> None:
        # LONG @ entry 1.175, SL 1.170, BOS_DOWN at 1.172. New SL =
        # 1.172 - 2*spread_pip(=0.8)*0.0001 = 1.17184.
        new_sl = _compute_new_sl(
            side="LONG",
            current_sl=1.17000,
            bos_price=1.17200,
            spread_pips=0.8,
            pair="EUR_USD",
        )
        self.assertIsNotNone(new_sl)
        # Tighter than 1.17000 → higher value for LONG SL.
        self.assertGreater(new_sl, 1.17000)
        self.assertLess(new_sl, 1.17200)

    def test_short_tightens_sl_above_bos_plus_buffer(self) -> None:
        new_sl = _compute_new_sl(
            side="SHORT",
            current_sl=114.500,
            bos_price=114.000,
            spread_pips=1.6,
            pair="AUD_JPY",
        )
        self.assertIsNotNone(new_sl)
        self.assertLess(new_sl, 114.500)
        self.assertGreater(new_sl, 114.000)

    def test_long_refuses_to_widen_sl(self) -> None:
        # Current SL 1.17300 (very tight). BOS_DOWN at 1.17200 would
        # give new SL ~1.17184, which is LOWER than 1.17300 = widening.
        # Must return None.
        new_sl = _compute_new_sl(
            side="LONG",
            current_sl=1.17300,
            bos_price=1.17200,
            spread_pips=0.8,
            pair="EUR_USD",
        )
        self.assertIsNone(new_sl)

    def test_short_refuses_to_widen_sl(self) -> None:
        new_sl = _compute_new_sl(
            side="SHORT",
            current_sl=113.800,
            bos_price=114.000,
            spread_pips=1.6,
            pair="AUD_JPY",
        )
        self.assertIsNone(new_sl)

    def test_no_spread_data_returns_none(self) -> None:
        # AGENT_CONTRACT §3.5: missing market data → no silent fallback.
        new_sl = _compute_new_sl(
            side="LONG",
            current_sl=1.17000,
            bos_price=1.17200,
            spread_pips=None,
            pair="EUR_USD",
        )
        self.assertIsNone(new_sl)


class ApplyTrailingSlTest(unittest.TestCase):
    """End-to-end behavior + existing-position invariants."""

    def setUp(self) -> None:
        self._prior_thresh = os.environ.pop("QR_TRAILING_SL_FROM_TRADE_ID", None)

    def tearDown(self) -> None:
        if self._prior_thresh is None:
            os.environ.pop("QR_TRAILING_SL_FROM_TRADE_ID", None)
        else:
            os.environ["QR_TRAILING_SL_FROM_TRADE_ID"] = self._prior_thresh

    def test_position_without_sl_is_never_touched(self) -> None:
        # CRITICAL invariant: SL-free positions (including every position
        # open before 2026-05-13) have stop_loss=None and MUST be skipped.
        snapshot = _snapshot(positions=[_long_position(sl=None)])
        charts = _pair_charts(EUR_USD="M15(struct=BOS_DOWN@1.17200); H1(struct=BOS_DOWN@1.17000)")
        client = MagicMock()
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        self.assertEqual(updates, [])
        client.replace_trade_dependent_orders.assert_not_called()

    def test_position_with_sl_and_thesis_reverse_bos_tightens(self) -> None:
        snapshot = _snapshot(positions=[_long_position(sl=1.17000)])
        charts = _pair_charts(EUR_USD="M15(struct=BOS_DOWN@1.17200); H1(struct=BOS_UP@1.18000)")
        client = MagicMock()
        client.replace_trade_dependent_orders.return_value = {}
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        self.assertEqual(len(updates), 1)
        self.assertTrue(updates[0].applied)
        self.assertGreater(updates[0].new_sl, updates[0].old_sl)
        client.replace_trade_dependent_orders.assert_called_once()

    def test_thesis_aligned_bos_does_not_update(self) -> None:
        # LONG position + M15/H1 both print BOS_UP (aligned, NOT against).
        # No trailing update.
        snapshot = _snapshot(positions=[_long_position(sl=1.17000)])
        charts = _pair_charts(EUR_USD="M15(struct=BOS_UP@1.17800); H1(struct=BOS_UP@1.18000)")
        client = MagicMock()
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        self.assertEqual(updates, [])
        client.replace_trade_dependent_orders.assert_not_called()

    def test_dry_run_does_not_call_broker(self) -> None:
        snapshot = _snapshot(positions=[_short_position(sl=114.500)])
        charts = _pair_charts(AUD_JPY="M15(struct=BOS_UP@114.000); H1(struct=BOS_UP@114.100)")
        client = MagicMock()
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
            dry_run=True,
        )
        self.assertEqual(len(updates), 1)
        self.assertFalse(updates[0].applied)
        client.replace_trade_dependent_orders.assert_not_called()

    def test_from_trade_id_threshold_filters_old_positions(self) -> None:
        # Configure threshold so only trade_id > 9000000 is eligible.
        os.environ["QR_TRAILING_SL_FROM_TRADE_ID"] = "9000000"
        old = _long_position(sl=1.17000, trade_id="8000000")  # below threshold
        new = _long_position(sl=1.17000, trade_id="9000001")  # above threshold
        snapshot = _snapshot(positions=[old, new])
        charts = _pair_charts(EUR_USD="M15(struct=BOS_DOWN@1.17200); H1(struct=BOS_UP@1.18000)")
        client = MagicMock()
        client.replace_trade_dependent_orders.return_value = {}
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].trade_id, "9000001")

    def test_manual_owner_position_never_touched(self) -> None:
        # Manual/unknown-owner positions belong to the operator; the
        # trader code path must not touch them.
        from quant_rabbit.models import BrokerPosition as BP
        manual = BP(
            trade_id="MANUAL-1",
            pair="EUR_USD",
            side=Side.LONG,
            units=1000,
            entry_price=1.17500,
            unrealized_pl_jpy=0.0,
            take_profit=1.18000,
            stop_loss=1.17000,
            owner=Owner.UNKNOWN,
        )
        snapshot = _snapshot(positions=[manual])
        charts = _pair_charts(EUR_USD="M15(struct=BOS_DOWN@1.17200); H1(struct=BOS_DOWN@1.17000)")
        client = MagicMock()
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        self.assertEqual(updates, [])

    def test_broker_error_does_not_raise(self) -> None:
        # OANDA replace_trade_dependent_orders may fail mid-flight; the
        # loop must continue, recording applied=False for the failure.
        snapshot = _snapshot(positions=[_long_position(sl=1.17000)])
        charts = _pair_charts(EUR_USD="M15(struct=BOS_DOWN@1.17200); H1(struct=BOS_UP@1.18000)")
        client = MagicMock()
        client.replace_trade_dependent_orders.side_effect = RuntimeError("broker timeout")
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        self.assertEqual(len(updates), 1)
        self.assertFalse(updates[0].applied)


class TimeframeListTest(unittest.TestCase):
    def test_trailing_timeframes_documented(self) -> None:
        # Pin the documented operator-relevant band: M15 / M30 / H1.
        # M5 / M1 are too noisy; H4 / D are too slow for entry-time
        # trailing. Changing this requires explicit operator review.
        self.assertEqual(TRAILING_TIMEFRAMES, ("M15", "M30", "H1"))


if __name__ == "__main__":
    unittest.main()

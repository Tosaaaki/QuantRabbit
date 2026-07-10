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
        # Tuple now carries close_confirmed flag (added 2026-05-13).
        # Events without `:wick` suffix are close-confirmed = True.
        self.assertEqual(events.get("M15"), ("BOS", "DOWN", 1.17200, True))
        self.assertEqual(events.get("H1"), ("CHOCH", "UP", 1.18000, True))
        # All TFs parsed; consumer filters by TRAILING_TIMEFRAMES.
        self.assertIn("M5", events)
        self.assertIn("H4", events)

    def test_parses_wick_suffix_as_not_close_confirmed(self) -> None:
        # 2026-05-13 wick filter: trailing must distinguish a real break
        # from a stop-hunt sweep so it does not tighten SL on a
        # candle that closed back inside the prior range.
        events = _parse_struct_events(
            "AUD_JPY; M15(struct=BOS_UP@114.1460:wick); "
            "H4(struct=BOS_UP@113.5870)"
        )
        self.assertEqual(events.get("M15"), ("BOS", "UP", 114.146, False))
        self.assertEqual(events.get("H4"), ("BOS", "UP", 113.587, True))

    def test_returns_empty_for_missing_string(self) -> None:
        self.assertEqual(_parse_struct_events(""), {})
        self.assertEqual(_parse_struct_events("no struct here"), {})


class ComputeNewSlTest(unittest.TestCase):
    def test_long_tightens_sl_below_bos_minus_buffer(self) -> None:
        # LONG entry 1.175 (well above the BOS), SL 1.170, BOS_DOWN at
        # 1.172. New SL = 1.172 - 2*spread_pip(=0.8)*0.0001 = 1.17184.
        # M15 ATR 5 pips → noise floor = entry - 5×1.5 pip = 1.1675;
        # candidate 1.17184 is BELOW that floor (good) so the update
        # proceeds.
        new_sl = _compute_new_sl(
            side="LONG",
            entry_price=1.17500,
            current_sl=1.17000,
            bos_price=1.17200,
            spread_pips=0.8,
            m15_atr_pips=5.0,
            pair="EUR_USD",
        )
        self.assertIsNotNone(new_sl)
        # Tighter than 1.17000 → higher value for LONG SL.
        self.assertGreater(new_sl, 1.17000)
        self.assertLess(new_sl, 1.17200)

    def test_short_tightens_sl_above_bos_plus_buffer(self) -> None:
        # SHORT entry 114.500, BOS_UP at 114.000 (well below entry).
        # M15 ATR 10 pips → noise floor = entry + 10×1.5 pip = 114.65;
        # candidate ~114.032 is ABOVE that floor? No — for SHORT the
        # candidate is ABOVE BOS, and the floor demands candidate be
        # ≥ entry + min_distance = 114.65. Candidate ~114.032 is BELOW
        # 114.65 so the noise floor would BLOCK the update. To exercise
        # the original tighten path with the floor active, place entry
        # closer to the BOS so the buffer-derived candidate clears the
        # floor.
        new_sl = _compute_new_sl(
            side="SHORT",
            entry_price=113.700,
            current_sl=114.500,
            bos_price=114.000,
            spread_pips=1.6,
            m15_atr_pips=5.0,
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
            entry_price=1.17500,
            current_sl=1.17300,
            bos_price=1.17200,
            spread_pips=0.8,
            m15_atr_pips=5.0,
            pair="EUR_USD",
        )
        self.assertIsNone(new_sl)

    def test_short_refuses_to_widen_sl(self) -> None:
        new_sl = _compute_new_sl(
            side="SHORT",
            entry_price=113.700,
            current_sl=113.800,
            bos_price=114.000,
            spread_pips=1.6,
            m15_atr_pips=5.0,
            pair="AUD_JPY",
        )
        self.assertIsNone(new_sl)

    def test_no_spread_data_returns_none(self) -> None:
        # AGENT_CONTRACT §3.5: missing market data → no silent fallback.
        new_sl = _compute_new_sl(
            side="LONG",
            entry_price=1.17500,
            current_sl=1.17000,
            bos_price=1.17200,
            spread_pips=None,
            m15_atr_pips=5.0,
            pair="EUR_USD",
        )
        self.assertIsNone(new_sl)

    def test_short_noise_floor_blocks_tighten_into_entry_band(self) -> None:
        # 2026-05-13 AUD_JPY 470989 regression. Entry 114.200, BOS_UP at
        # 114.146 with M15 ATR 5 pips → floor = 114.200 + 5×1.5 pip =
        # 114.275. Candidate = 114.146 + 2*spread_pip(1.6)*0.01 = 114.178.
        # 114.178 < 114.275 → trailing is INSIDE the noise band → return
        # None (no tighten). This is the structural fix that stops the
        # SL-tighten-to-entry+0.4pip pattern that the broker stopped out
        # twice in 16 minutes.
        new_sl = _compute_new_sl(
            side="SHORT",
            entry_price=114.200,
            current_sl=114.629,
            bos_price=114.146,
            spread_pips=1.6,
            m15_atr_pips=5.0,
            pair="AUD_JPY",
        )
        self.assertIsNone(new_sl)

    def test_long_noise_floor_blocks_tighten_into_entry_band(self) -> None:
        # Symmetric LONG case: entry 1.17500, BOS_DOWN at 1.17480 with
        # M15 ATR 3 pips → floor = 1.17500 - 3×1.5 pip = 1.17455.
        # Candidate = 1.17480 - 2*0.8 pip = 1.17464. 1.17464 > 1.17455
        # (noise band) → return None.
        new_sl = _compute_new_sl(
            side="LONG",
            entry_price=1.17500,
            current_sl=1.17000,
            bos_price=1.17480,
            spread_pips=0.8,
            m15_atr_pips=3.0,
            pair="EUR_USD",
        )
        self.assertIsNone(new_sl)

    def test_noise_floor_degrades_to_no_floor_when_atr_unavailable(self) -> None:
        # If chart_story format drift hides the M15 ATR, the floor
        # collapses to 0 and the function falls back to the prior
        # buffer-only behaviour — better than crashing the cycle.
        new_sl = _compute_new_sl(
            side="SHORT",
            entry_price=114.200,
            current_sl=114.629,
            bos_price=114.146,
            spread_pips=1.6,
            m15_atr_pips=None,
            pair="AUD_JPY",
        )
        # With no floor, the candidate is the BOS + buffer = 114.178,
        # which is tighter than 114.629 → the update goes through.
        self.assertIsNotNone(new_sl)
        self.assertLess(new_sl, 114.629)


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

    def test_predictive_scout_stop_is_not_trailed(self) -> None:
        scout = BrokerPosition(
            trade_id="scout-1",
            pair="EUR_USD",
            side=Side.LONG,
            units=1000,
            entry_price=1.17500,
            unrealized_pl_jpy=0.0,
            take_profit=1.18000,
            stop_loss=1.17000,
            owner=Owner.TRADER,
            raw={
                "tradeClientExtensions": {
                    "comment": "qr-vnext role=BIDASK_REPLAY_CONTRARIAN_SCOUT vehicle=psv-test"
                }
            },
        )
        snapshot = _snapshot(positions=[scout])
        charts = _pair_charts(EUR_USD="M15(struct=BOS_DOWN@1.17200); H1(struct=BOS_UP@1.18000)")
        client = MagicMock()

        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )

        self.assertEqual(updates, [])
        client.replace_trade_dependent_orders.assert_not_called()

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

    def test_wick_only_event_does_not_trigger_trailing(self) -> None:
        # 2026-05-13 wick filter end-to-end: a `:wick` suffix on the
        # M15 BOS event must NOT fire trailing, even if it is in the
        # right direction (counter to position side).
        snapshot = _snapshot(positions=[_short_position(sl=114.500)])
        charts = _pair_charts(
            AUD_JPY="M15(struct=BOS_UP@114.000:wick); H1(struct=CHOCH_DOWN@113.900)"
        )
        client = MagicMock()
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        # M15 is wick-only AND counter-direction; H1 is aligned with
        # SHORT thesis (not against). No trailing fires.
        self.assertEqual(updates, [])
        client.replace_trade_dependent_orders.assert_not_called()

    def test_noise_floor_blocks_tighten_when_bos_near_entry(self) -> None:
        # 2026-05-13 AUD_JPY 470989/470997 end-to-end regression: SHORT
        # entry 114.200, broker SL 114.629 (42 pip), BOS_UP printed at
        # 114.146 close-confirmed AND M15 ATR is 5p (low Tokyo session).
        # Without the noise floor, trailing tightens SL to ~114.178
        # (BOS + 2×spread buffer), entry+0.4 pip; routine bid noise then
        # taps it. With the floor, the candidate is REJECTED because it
        # lands inside `entry + 1.5 × M15_ATR = entry + 7.5p = 114.275`
        # noise band. No trailing fires.
        regression_short = BrokerPosition(
            trade_id="470984",
            pair="AUD_JPY",
            side=Side.SHORT,
            units=13000,
            entry_price=114.200,
            unrealized_pl_jpy=-30.0,
            take_profit=112.617,
            stop_loss=114.629,
            owner=Owner.TRADER,
        )
        snapshot = _snapshot(positions=[regression_short])
        # chart_story includes ATR=5.0p in the M15 segment so the floor
        # has a market-derived distance to compare against.
        charts = _pair_charts(
            AUD_JPY=(
                "M15(RANGE, ATR=5.0p struct=BOS_UP@114.146); "
                "H1(RANGE, ATR=12.0p struct=BOS_UP@114.000)"
            )
        )
        client = MagicMock()
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=charts,
            broker_client=client,
        )
        # M15 trips first by TRAILING_TIMEFRAMES order; candidate is
        # rejected by the noise floor; the loop breaks on the first
        # matching event regardless of whether it produces an SL
        # update. The position keeps its original 114.629 SL.
        self.assertEqual(updates, [])
        client.replace_trade_dependent_orders.assert_not_called()


class TimeframeListTest(unittest.TestCase):
    def test_trailing_timeframes_documented(self) -> None:
        # Pin the documented operator-relevant band: M15 / M30 / H1.
        # M5 / M1 are too noisy; H4 / D are too slow for entry-time
        # trailing. Changing this requires explicit operator review.
        self.assertEqual(TRAILING_TIMEFRAMES, ("M15", "M30", "H1"))


if __name__ == "__main__":
    unittest.main()

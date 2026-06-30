from __future__ import annotations

import unittest
from datetime import datetime, timezone

from quant_rabbit.models import AccountSummary, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.operator_manual import (
    OPERATOR_ALPHA_CANDIDATE,
    OPERATOR_MANUAL,
    OPERATOR_MANUAL_POSITION_PACKET,
    classify_operator_manual_snapshot,
    major_figure_fade_thesis_state,
    operator_manual_position_packets,
)


class OperatorManualClassificationTest(unittest.TestCase):
    def test_confirmed_unknown_usdjpy_short_becomes_operator_manual_packet(self) -> None:
        now = datetime(2026, 6, 30, 1, 0, tzinfo=timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="manual-usdjpy-1",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=22_000,
                    entry_price=161.84,
                    unrealized_pl_jpy=-12_300.0,
                    owner=Owner.UNKNOWN,
                    raw={"clientExtensions": {"tag": ""}},
                ),
            ),
            quotes={"USD_JPY": Quote("USD_JPY", bid=161.92, ask=161.93, timestamp_utc=now)},
            account=AccountSummary(
                nav_jpy=250_000.0,
                balance_jpy=262_300.0,
                unrealized_pl_jpy=-12_300.0,
                margin_used_jpy=142_419.2,
                margin_available_jpy=107_580.8,
                fetched_at_utc=now,
            ),
        )

        classified = classify_operator_manual_snapshot(
            snapshot,
            confirmations=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 22_000,
                    "owner_confirmed": True,
                    "alpha_classification": OPERATOR_ALPHA_CANDIDATE,
                    "thesis": "162.00 historical/intervention-risk fade",
                    "invalidation": "accepted trade above 162.00 major figure",
                    "harvest_trigger": "harvest after rejection from 162.00",
                    "harvest_zone": "below 162.00 after rejection",
                    "major_figure": 162.0,
                }
            ],
        )

        position = classified.positions[0]
        self.assertEqual(position.owner, Owner.OPERATOR_MANUAL)
        self.assertEqual(position.raw["operator_manual_position"]["packet_type"], OPERATOR_MANUAL_POSITION_PACKET)

        packets = operator_manual_position_packets(classified)
        self.assertEqual(len(packets), 1)
        packet = packets[0]
        self.assertEqual(packet["classification"], OPERATOR_MANUAL)
        self.assertEqual(packet["alpha_classification"], OPERATOR_ALPHA_CANDIDATE)
        self.assertEqual(packet["pair"], "USD_JPY")
        self.assertEqual(packet["side"], "SHORT")
        self.assertEqual(packet["units"], 22_000)
        self.assertEqual(packet["pip_value_jpy_per_pip"], 220.0)
        self.assertEqual(packet["thesis_state"], "ALIVE")
        self.assertIn("red P/L", packet["exact_invalidation_evidence"])
        self.assertTrue(packet["blocks_fresh_jpy_adds"])

    def test_system_lane_receipt_prevents_operator_manual_reclassification(self) -> None:
        now = datetime(2026, 6, 30, 1, 0, tzinfo=timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="system-lane",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=22_000,
                    entry_price=161.84,
                    owner=Owner.UNKNOWN,
                    raw={"clientExtensions": {"tag": "trader", "comment": "lane receipt"}},
                ),
            ),
        )

        classified = classify_operator_manual_snapshot(
            snapshot,
            confirmations=[{"pair": "USD_JPY", "side": "SHORT", "units": 22_000, "owner_confirmed": True}],
        )

        self.assertEqual(classified.positions[0].owner, Owner.UNKNOWN)

    def test_major_figure_fade_requires_accepted_break_not_wick_or_red_upl(self) -> None:
        alive = major_figure_fade_thesis_state(
            side=Side.SHORT,
            quote=Quote("USD_JPY", bid=161.91, ask=161.92),
            unrealized_pl_jpy=-18_000.0,
            major_figure=162.0,
        )
        wick = major_figure_fade_thesis_state(
            side=Side.SHORT,
            quote=Quote("USD_JPY", bid=161.99, ask=161.995),
            unrealized_pl_jpy=-18_000.0,
            major_figure=162.0,
            wick_above=True,
        )
        accepted = major_figure_fade_thesis_state(
            side=Side.SHORT,
            quote=Quote("USD_JPY", bid=162.08, ask=162.09),
            unrealized_pl_jpy=-18_000.0,
            major_figure=162.0,
            accepted_break=True,
        )

        self.assertEqual(alive["state"], "ALIVE")
        self.assertEqual(wick["state"], "WOUNDED")
        self.assertEqual(accepted["state"], "INVALIDATED")


if __name__ == "__main__":
    unittest.main()

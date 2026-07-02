from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import AccountSummary, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.operator_manual import (
    OPERATOR_ALPHA_CANDIDATE,
    OPERATOR_MANUAL,
    OPERATOR_MANUAL_POSITION_PACKET,
    classify_operator_manual_snapshot,
    load_operator_manual_confirmations,
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

    def test_operator_confirmed_eur_usd_unknown_trade_becomes_operator_manual_only_for_trade_id(self) -> None:
        now = datetime(2026, 7, 2, 7, 34, tzinfo=timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="472987",
                    pair="EUR_USD",
                    side=Side.SHORT,
                    units=30_000,
                    entry_price=1.14048,
                    unrealized_pl_jpy=-922.0941,
                    take_profit=1.13800,
                    stop_loss=None,
                    owner=Owner.UNKNOWN,
                    raw={"currentUnits": "-30000", "price": "1.14048"},
                ),
                BrokerPosition(
                    trade_id="other-eurusd",
                    pair="EUR_USD",
                    side=Side.SHORT,
                    units=30_000,
                    entry_price=1.14010,
                    owner=Owner.UNKNOWN,
                    raw={"currentUnits": "-30000", "price": "1.14010"},
                ),
            ),
            quotes={"EUR_USD": Quote("EUR_USD", bid=1.14070, ask=1.14078, timestamp_utc=now)},
        )

        classified = classify_operator_manual_snapshot(
            snapshot,
            confirmations=[
                {
                    "trade_id": "472987",
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30_000,
                    "owner_confirmed": True,
                    "operator_decision": "OPERATOR_CONFIRMED_MANUAL_OWNED",
                    "management_intent": "KEEP",
                    "reason": "operator explicitly confirmed manual EUR_USD should remain open",
                    "operator_confirmation_source": "chat_operator_confirmation",
                    "no_live_side_effects": True,
                    "system_pl_counted": False,
                    "same_theme_auto_add_allowed": False,
                    "loss_side_auto_close_allowed": False,
                    "auto_sl_attach_allowed": False,
                    "auto_tp_modify_allowed": False,
                    "thesis": "operator-confirmed manual EUR_USD short",
                }
            ],
        )

        owners = {position.trade_id: position.owner for position in classified.positions}
        self.assertEqual(owners["472987"], Owner.OPERATOR_MANUAL)
        self.assertEqual(owners["other-eurusd"], Owner.UNKNOWN)
        packet = classified.positions[0].raw["operator_manual_position"]
        self.assertEqual(packet["operator_decision"], "OPERATOR_CONFIRMED_MANUAL_OWNED")
        self.assertEqual(packet["management_intent"], "KEEP")
        self.assertEqual(packet["operator_confirmation_source"], "chat_operator_confirmation")
        self.assertFalse(packet["system_pl_counted"])
        self.assertFalse(packet["same_theme_auto_add_allowed"])
        self.assertFalse(packet["loss_side_auto_close_allowed"])
        self.assertFalse(packet["auto_sl_attach_allowed"])
        self.assertFalse(packet["auto_tp_modify_allowed"])

        packets = operator_manual_position_packets(classified)
        eur_packet = next(item for item in packets if item["pair"] == "EUR_USD")
        self.assertEqual(eur_packet["trade_ids"], ["472987"])
        self.assertEqual(eur_packet["operator_decision"], "OPERATOR_CONFIRMED_MANUAL_OWNED")
        self.assertFalse(eur_packet["system_pl_counted"])

    def test_operator_review_artifact_row_loads_as_manual_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            review_path = root / "guardian_receipt_operator_review.json"
            review_path.write_text(
                json.dumps(
                    {
                        "operator_position_reviews": [
                            {
                                "trade_id": "472987",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "units": 30_000,
                                "owner": "OPERATOR_MANUAL",
                                "operator_decision": "OPERATOR_CONFIRMED_MANUAL_OWNED",
                                "management_intent": "KEEP",
                                "operator_confirmation_source": "chat_operator_confirmation",
                                "system_pl_counted": False,
                                "same_theme_auto_add_allowed": False,
                                "loss_side_auto_close_allowed": False,
                                "auto_sl_attach_allowed": False,
                                "auto_tp_modify_allowed": False,
                            },
                            {
                                "trade_id": "skip-me",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "owner": "UNKNOWN",
                                "operator_decision": "OPERATOR_CONFIRMED_NO_ACTION",
                            },
                        ],
                    }
                )
            )

            rows = load_operator_manual_confirmations(
                root / "operator_manual_positions.json",
                operator_review_path=review_path,
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["trade_id"], "472987")
        self.assertTrue(row["operator_confirmed"])
        self.assertTrue(row["owner_confirmed"])
        self.assertEqual(row["classification"], OPERATOR_MANUAL)
        self.assertEqual(row["operator_confirmation_source"], "chat_operator_confirmation")
        self.assertFalse(row["system_pl_counted"])
        self.assertFalse(row["same_theme_auto_add_allowed"])
        self.assertFalse(row["loss_side_auto_close_allowed"])
        self.assertFalse(row["auto_sl_attach_allowed"])
        self.assertFalse(row["auto_tp_modify_allowed"])

    def test_confirmed_split_tranche_classifies_oldest_units_when_extra_unknown_exists(self) -> None:
        now = datetime(2026, 6, 30, 3, 30, tzinfo=timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="new-extra-2",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=1_000,
                    entry_price=162.157,
                    unrealized_pl_jpy=-77.0,
                    owner=Owner.UNKNOWN,
                    raw={"openTime": "2026-06-30T03:06:05.083752860Z"},
                ),
                BrokerPosition(
                    trade_id="new-extra-1",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=1_000,
                    entry_price=162.139,
                    unrealized_pl_jpy=-95.0,
                    owner=Owner.UNKNOWN,
                    raw={"openTime": "2026-06-30T02:49:26.217785906Z"},
                ),
                BrokerPosition(
                    trade_id="confirmed-4",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=1_000,
                    entry_price=162.146,
                    unrealized_pl_jpy=-88.0,
                    owner=Owner.UNKNOWN,
                    raw={"openTime": "2026-06-30T02:19:45.445812800Z"},
                ),
                BrokerPosition(
                    trade_id="confirmed-3",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=1_000,
                    entry_price=162.156,
                    unrealized_pl_jpy=-78.0,
                    owner=Owner.UNKNOWN,
                    raw={"openTime": "2026-06-30T02:01:39.173639648Z"},
                ),
                BrokerPosition(
                    trade_id="confirmed-2",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=5_000,
                    entry_price=161.892,
                    unrealized_pl_jpy=-1710.0,
                    owner=Owner.UNKNOWN,
                    raw={"openTime": "2026-06-29T14:02:12.760635166Z"},
                ),
                BrokerPosition(
                    trade_id="confirmed-1",
                    pair="USD_JPY",
                    side=Side.SHORT,
                    units=15_000,
                    entry_price=161.938,
                    unrealized_pl_jpy=-4440.0,
                    owner=Owner.UNKNOWN,
                    raw={"openTime": "2026-06-29T13:47:34.536835380Z"},
                ),
            ),
            quotes={"USD_JPY": Quote("USD_JPY", bid=162.228, ask=162.236, timestamp_utc=now)},
        )

        classified = classify_operator_manual_snapshot(
            snapshot,
            confirmations=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 22_000,
                    "owner_confirmed": True,
                    "thesis": "162.00 historical/intervention-risk fade",
                    "major_figure": 162.0,
                }
            ],
        )

        owners = {position.trade_id: position.owner for position in classified.positions}
        self.assertEqual(owners["confirmed-1"], Owner.OPERATOR_MANUAL)
        self.assertEqual(owners["confirmed-2"], Owner.OPERATOR_MANUAL)
        self.assertEqual(owners["confirmed-3"], Owner.OPERATOR_MANUAL)
        self.assertEqual(owners["confirmed-4"], Owner.OPERATOR_MANUAL)
        self.assertEqual(owners["new-extra-1"], Owner.UNKNOWN)
        self.assertEqual(owners["new-extra-2"], Owner.UNKNOWN)

        packets = operator_manual_position_packets(classified)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0]["units"], 22_000)
        self.assertEqual(packets[0]["unrealized_pl_jpy"], -6316.0)
        self.assertEqual(
            packets[0]["trade_ids"],
            ["confirmed-4", "confirmed-3", "confirmed-2", "confirmed-1"],
        )

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

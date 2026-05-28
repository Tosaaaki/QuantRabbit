from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_rabbit.models import BrokerPosition, Owner, Side
from quant_rabbit.strategy.entry_thesis_ledger import EntryThesis, record_entry_thesis
from quant_rabbit.strategy.position_thesis_validator import (
    PositionThesisAssessment,
    _apply_entry_invalidation_overrides,
    _reconcile_same_pair_hedges,
)


def _assessment(trade_id: str, side: str, verdict: str = "EXTEND") -> PositionThesisAssessment:
    return PositionThesisAssessment(
        trade_id=trade_id,
        pair="EUR_USD",
        side=side,
        pattern_score=10.0,
        projection_score=15.0,
        correlation_score=0.0,
        path_score=0.0,
        reversal_against=False,
        aggregate_score=25.0,
        verdict=verdict,
        rationale_lines=("synthetic detector support",),
    )


def _up_tech_chart() -> dict:
    return {
        "views": [
            {
                "granularity": tf,
                "regime": "TREND_UP",
                "indicators": {
                    "rsi_14": 70.0,
                    "macd_hist": 0.0002,
                    "supertrend_dir": 1,
                    "ichimoku_cloud_pos": 1,
                    "plus_di_14": 35.0,
                    "minus_di_14": 10.0,
                },
                "structure": {"last_event": {"kind": "CHOCH_UP", "close_confirmed": True}},
            }
            for tf in ("M5", "M15")
        ]
    }


class PositionThesisValidatorTest(unittest.TestCase):
    def test_same_pair_opposite_extends_are_suppressed_to_hold(self) -> None:
        positions = [
            BrokerPosition("long-1", "EUR_USD", Side.LONG, 7000, 1.16, owner=Owner.TRADER),
            BrokerPosition("short-1", "EUR_USD", Side.SHORT, 2000, 1.16, owner=Owner.TRADER),
        ]

        reconciled = _reconcile_same_pair_hedges(
            [_assessment("long-1", "LONG"), _assessment("short-1", "SHORT")],
            positions,
        )

        self.assertEqual([item.verdict for item in reconciled], ["HOLD", "HOLD"])
        self.assertTrue(all("same-pair trader hedge context" in " ".join(item.context_notes) for item in reconciled))
        self.assertTrue(all("EXTEND suppressed" in " ".join(item.context_notes) for item in reconciled))

    def test_single_active_side_keeps_extend(self) -> None:
        positions = [
            BrokerPosition("long-1", "EUR_USD", Side.LONG, 7000, 1.16, owner=Owner.TRADER),
        ]

        reconciled = _reconcile_same_pair_hedges([_assessment("long-1", "LONG")], positions)

        self.assertEqual(reconciled[0].verdict, "EXTEND")
        self.assertEqual(reconciled[0].context_notes, ())

    def test_entry_invalidation_overrides_hold_to_review_close(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record_entry_thesis(
                EntryThesis(
                    timestamp_utc="2026-05-28T06:55:31Z",
                    trade_id="short-1",
                    pair="EUR_USD",
                    side="SHORT",
                    entry_price=1.1609,
                    forecast_direction="DOWN",
                    forecast_confidence=0.69,
                    regime="RANGE",
                    invalidation_price=1.16097,
                    target_price=1.16019,
                    key_drivers=[],
                ),
                root,
            )
            positions = [
                BrokerPosition("short-1", "EUR_USD", Side.SHORT, 7000, 1.1609, owner=Owner.TRADER),
            ]

            overridden = _apply_entry_invalidation_overrides(
                [_assessment("short-1", "SHORT", verdict="HOLD")],
                positions,
                quotes_by_pair={"EUR_USD": {"ask": 1.16325, "bid": 1.16317}},
                pair_charts_full={"EUR_USD": _up_tech_chart()},
                data_root=root,
            )

            self.assertEqual(overridden[0].verdict, "REVIEW_CLOSE")
            self.assertIn("invalidation hit", " ".join(overridden[0].context_notes))


if __name__ == "__main__":
    unittest.main()

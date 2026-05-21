from __future__ import annotations

import unittest

from quant_rabbit.models import BrokerPosition, Owner, Side
from quant_rabbit.strategy.position_thesis_validator import (
    PositionThesisAssessment,
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


if __name__ == "__main__":
    unittest.main()

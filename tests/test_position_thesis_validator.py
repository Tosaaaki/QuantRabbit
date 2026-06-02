from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_rabbit.models import BrokerPosition, Owner, Side
from quant_rabbit.strategy.entry_thesis_ledger import EntryThesis, record_entry_thesis
from quant_rabbit.strategy.position_thesis_validator import (
    PositionThesisAssessment,
    _apply_entry_invalidation_overrides,
    _chart_alignment_score,
    _reconcile_same_pair_hedges,
)


def _assessment(
    trade_id: str,
    side: str,
    verdict: str = "EXTEND",
    *,
    pattern_score: float = 10.0,
    projection_score: float = 15.0,
    aggregate_score: float | None = None,
) -> PositionThesisAssessment:
    aggregate = aggregate_score if aggregate_score is not None else pattern_score + projection_score
    return PositionThesisAssessment(
        trade_id=trade_id,
        pair="EUR_USD",
        side=side,
        pattern_score=pattern_score,
        projection_score=projection_score,
        correlation_score=0.0,
        path_score=0.0,
        technical_score=0.0,
        reversal_against=False,
        aggregate_score=aggregate,
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


def _broad_down_tech_chart() -> dict:
    return {
        "confluence": {
            "score_balance": "SHORT_LEAN",
            "score_gap": -0.93,
            "price_percentile_24h": 0.01,
            "range_24h_sigma_multiple": 4.3,
        },
        "views": [
            {
                "granularity": tf,
                "long_bias": 0.0,
                "short_bias": 1.0,
                "regime": "TREND_DOWN",
                "indicators": {
                    "rsi_14": 28.0,
                    "macd_hist": -0.0002,
                    "supertrend_dir": -1,
                    "ichimoku_cloud_pos": -1,
                    "plus_di_14": 10.0,
                    "minus_di_14": 35.0,
                },
                "structure": {"last_event": {"kind": "BOS_DOWN", "close_confirmed": True}},
            }
            for tf in ("M5", "M15", "M30", "H1")
        ],
    }


class PositionThesisValidatorTest(unittest.TestCase):
    def test_chart_alignment_scores_strong_opposite_panel_against_position(self) -> None:
        long_score, long_reasons = _chart_alignment_score(_broad_down_tech_chart(), "LONG")
        short_score, short_reasons = _chart_alignment_score(_broad_down_tech_chart(), "SHORT")

        self.assertLess(long_score, -20.0)
        self.assertGreater(short_score, 20.0)
        self.assertTrue(any("SHORT_LEAN" in reason for reason in long_reasons))
        self.assertTrue(any("operating TF" in reason for reason in short_reasons))

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
            self.assertIn("post-close re-entry discipline", " ".join(overridden[0].context_notes))

    def test_missing_invalidation_underwater_uses_entry_buffer_plus_technicals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            positions = [
                BrokerPosition(
                    "short-1",
                    "EUR_USD",
                    Side.SHORT,
                    1000,
                    1.16356,
                    unrealized_pl_jpy=-57.5,
                    owner=Owner.TRADER,
                ),
            ]

            overridden = _apply_entry_invalidation_overrides(
                [_assessment(
                    "short-1",
                    "SHORT",
                    verdict="HOLD",
                    pattern_score=0.0,
                    projection_score=0.0,
                )],
                positions,
                quotes_by_pair={"EUR_USD": {"ask": 1.16392, "bid": 1.16384}},
                pair_charts_full={"EUR_USD": _up_tech_chart()},
                data_root=Path(td),
            )

        self.assertEqual(overridden[0].verdict, "REVIEW_CLOSE")
        notes = " ".join(overridden[0].context_notes)
        self.assertIn("adverse technical loss", notes)
        self.assertIn("technical invalidation confirmed against SHORT", notes)
        self.assertIn("post-close re-entry discipline", notes)

    def test_missing_invalidation_loss_cut_deferred_when_recovery_stack_supports_position(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            positions = [
                BrokerPosition(
                    "short-1",
                    "EUR_USD",
                    Side.SHORT,
                    7000,
                    1.1609,
                    unrealized_pl_jpy=-4647.7,
                    owner=Owner.TRADER,
                ),
            ]

            overridden = _apply_entry_invalidation_overrides(
                [_assessment(
                    "short-1",
                    "SHORT",
                    verdict="EXTEND",
                    pattern_score=30.0,
                    projection_score=25.0,
                )],
                positions,
                quotes_by_pair={"EUR_USD": {"ask": 1.16506, "bid": 1.16498}},
                pair_charts_full={"EUR_USD": _up_tech_chart()},
                data_root=Path(td),
            )

        self.assertEqual(overridden[0].verdict, "HOLD")
        notes = " ".join(overridden[0].context_notes)
        self.assertIn("adverse technical loss", notes)
        self.assertIn("loss-cut deferred", notes)
        self.assertIn("current prediction stack still supports SHORT", notes)

    def test_missing_invalidation_loss_cut_not_deferred_when_broad_technicals_invalidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            positions = [
                BrokerPosition(
                    "long-1",
                    "EUR_USD",
                    Side.LONG,
                    4000,
                    1.1667,
                    unrealized_pl_jpy=-2692.1,
                    owner=Owner.TRADER,
                ),
            ]

            overridden = _apply_entry_invalidation_overrides(
                [_assessment(
                    "long-1",
                    "LONG",
                    verdict="EXTEND",
                    pattern_score=30.0,
                    projection_score=25.0,
                )],
                positions,
                quotes_by_pair={"EUR_USD": {"ask": 1.16258, "bid": 1.16250}},
                pair_charts_full={"EUR_USD": _broad_down_tech_chart()},
                data_root=Path(td),
            )

        self.assertEqual(overridden[0].verdict, "REVIEW_CLOSE")
        notes = " ".join(overridden[0].context_notes)
        self.assertIn("adverse technical loss", notes)
        self.assertIn("technical invalidation confirmed against LONG", notes)
        self.assertNotIn("loss-cut deferred", notes)

    def test_missing_invalidation_buffer_does_not_cut_profitable_position(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            positions = [
                BrokerPosition(
                    "short-1",
                    "EUR_USD",
                    Side.SHORT,
                    1000,
                    1.16356,
                    unrealized_pl_jpy=12.0,
                    owner=Owner.TRADER,
                ),
            ]

            overridden = _apply_entry_invalidation_overrides(
                [_assessment("short-1", "SHORT", verdict="HOLD")],
                positions,
                quotes_by_pair={"EUR_USD": {"ask": 1.16392, "bid": 1.16384}},
                pair_charts_full={"EUR_USD": _up_tech_chart()},
                data_root=Path(td),
            )

        self.assertEqual(overridden[0].verdict, "HOLD")
        self.assertEqual(overridden[0].context_notes, ())


if __name__ == "__main__":
    unittest.main()

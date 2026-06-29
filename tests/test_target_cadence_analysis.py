from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools import target_cadence_analysis as cadence


def _manual_payload() -> dict:
    return {
        "window": {"from": "2025-05-15T00:00:00Z", "to": "2025-07-15T00:00:00Z"},
        "exit_events": 3,
        "trades": [
            {
                "trade_id": "win-1",
                "pair": "USD_JPY",
                "units": 1000,
                "open_time": "2025-06-01T06:00:00Z",
                "close_time": "2025-06-01T08:00:00Z",
                "realized_pl": 10000.0,
                "close_reason": "TAKE_PROFIT_ORDER",
            },
            {
                "trade_id": "loss-1",
                "pair": "USD_JPY",
                "units": -1000,
                "open_time": "2025-06-02T12:00:00Z",
                "close_time": "2025-06-02T13:00:00Z",
                "realized_pl": -5000.0,
                "close_reason": "STOP_LOSS_ORDER",
            },
            {
                "trade_id": "win-2",
                "pair": "EUR_USD",
                "units": 1000,
                "open_time": "2025-06-03T00:00:00Z",
                "close_time": "2025-06-03T01:00:00Z",
                "realized_pl": 20000.0,
                "close_reason": "MARKET_ORDER_TRADE_CLOSE",
            },
        ],
        "analysis": {
            "daily_pl": {
                "2025-06-01": 10000.0,
                "2025-06-02": -5000.0,
                "2025-06-03": 20000.0,
            },
            "overall": {"net": 25000.0, "win_rate": 0.667},
            "by_pair": {},
            "by_side": {},
            "by_session_jst": {},
            "by_close_reason": {},
            "cash_flows": {
                "initial_balance": 100000.0,
                "transfers": [
                    {
                        "time": "2025-06-01T00:00:00Z",
                        "amount": 100000.0,
                        "balance": 100000.0,
                    }
                ],
                "transfer_adjusted_end_profit": 25000.0,
                "transfer_adjusted_end_return_pct": 25.0,
                "best_30d_funding_adjusted": {
                    "start_time": "2025-06-01T00:00:00+00:00",
                    "end_time": "2025-06-30T00:00:00+00:00",
                    "profit": 25000.0,
                    "return_pct": 25.0,
                },
            },
            "balance_start": {"time": "2025-06-01T00:00:00Z", "balance": 100000.0},
            "balance_end": {"time": "2025-06-03T23:59:00Z", "balance": 125000.0},
        },
    }


def _manual_context_payload() -> dict:
    return {
        "bounded_replay_profile": {
            "by_side_entry_location_24h": [],
            "by_h1_alignment": [],
            "by_session_jst": [],
            "by_close_reason": [],
        },
        "position_building_profile": {
            "bounded_by_build_type": [],
            "by_build_type": [],
            "adverse_adds": {},
            "contract": {"advisory_only": True},
        },
        "excluded_tail_profile": {},
    }


class TargetCadenceAnalysisTest(unittest.TestCase):
    def test_model_b_rolling_4x_uses_lower_daily_requirement_than_fixed_5pct(self) -> None:
        rows = cadence._daily_return_table(_manual_payload())
        models = cadence._target_models(rows)

        self.assertEqual(models["model_a_fixed_daily_5pct"]["required_average_daily_return_pct"], 5.0)
        self.assertAlmostEqual(
            models["model_b_rolling_30d_4x"]["required_average_daily_return_pct"],
            4.73,
            places=2,
        )
        self.assertGreater(
            models["model_a_fixed_daily_5pct"]["equivalent_30d_multiplier"],
            models["model_b_rolling_30d_4x"]["equivalent_30d_multiplier"],
        )

    def test_daily_table_estimates_returns_and_keeps_manual_as_teacher_data(self) -> None:
        rows = cadence._daily_return_table(_manual_payload())

        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0]["hit_10pct"])
        self.assertTrue(rows[1]["red"])
        self.assertEqual(rows[2]["dominant_pair"], "EUR_USD")

        breakdown = cadence._origin_breakdown(_manual_payload(), Path("/missing/execution_ledger.db"))
        self.assertEqual(breakdown["manual_2025"]["origin"], "OPERATOR_MANUAL")
        self.assertFalse(breakdown["manual_2025"]["system_profitability_counted"])
        self.assertEqual(breakdown["manual_2025"]["realized_pl_jpy"], 25000.0)

    def test_build_analysis_writes_pair_agnostic_shape_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_payload()))
            context = root / "context.json"
            context.write_text(json.dumps(_manual_context_payload()))
            precedent = root / "precedent.json"
            precedent.write_text(json.dumps({"precedent": {"sample": {"active_window": {}}}}))
            target_state = root / "target_state.json"
            target_state.write_text(json.dumps({}))
            args = argparse.Namespace(
                manual_history=manual,
                operator_precedent=precedent,
                manual_context=context,
                daily_target_state=target_state,
                execution_ledger=root / "missing.db",
            )

            target_payload, shape_payload = cadence.build_analysis(args)

            self.assertEqual(target_payload["recommendation"]["target_policy"], "Optimize for Model B: rolling 30-day 4x account growth.")
            ids = {row["pattern_id"] for row in shape_payload["lessons"]}
            self.assertIn("LOCATION_24H", ids)
            self.assertIn("MARGIN_CLOSEOUT_FAILURE", ids)
            self.assertIn("USD_JPY-only", " ".join(target_payload["recommendation"]["behaviors_to_block"]))


if __name__ == "__main__":
    unittest.main()

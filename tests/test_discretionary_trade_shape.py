from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.discretionary_trade_shape import evaluate_trade_shape_engine
from quant_rabbit.operator_precedent import build_operator_precedent_audit


def _manual_history_payload() -> dict:
    return {
        "transaction_count": 2309,
        "exit_events": 411,
        "closed_trades": 384,
        "reduced_trades": 27,
        "window": {"from": "2025-05-15T00:00:00Z", "to": "2025-07-15T00:00:00Z"},
        "analysis": {
            "overall": {
                "trades": 411,
                "net": 266815.9,
                "win_rate": 0.511,
                "payoff": 1.3,
                "median_hold_hours": 0.48,
                "expectancy": 649.2,
            },
            "by_pair": {
                "USD_JPY": {
                    "trades": 411,
                    "net": 266815.9,
                    "win_rate": 0.511,
                    "payoff": 1.3,
                    "median_hold_hours": 0.48,
                }
            },
            "by_side": {
                "LONG": {"trades": 240, "net": 351347.9, "win_rate": 0.596, "payoff": 1.78},
                "SHORT": {"trades": 171, "net": -84532.0, "win_rate": 0.392, "payoff": 1.14},
            },
            "by_session_jst": {
                "LONDON_AM": {"trades": 86, "net": 185804.8, "win_rate": 0.651, "payoff": 1.9},
                "NY_OVERLAP": {"trades": 188, "net": 88430.6, "win_rate": 0.473, "payoff": 1.52},
            },
            "by_close_reason": {
                "MARKET_ORDER_MARGIN_CLOSEOUT": {
                    "trades": 24,
                    "net": -217327.8,
                    "win_rate": 0.042,
                    "median_hold_hours": 12.38,
                }
            },
            "cash_flows": {
                "net_additional_transfers": 634172.0,
                "transfer_adjusted_peak_profit": 400557.6793,
                "transfer_adjusted_peak_return_pct": 200.28,
                "transfer_adjusted_end_profit": 269208.7038,
                "transfer_adjusted_end_return_pct": 134.6,
                "best_30d_funding_adjusted": {
                    "start_time": "2025-06-13T00:02:29.467604+00:00",
                    "end_time": "2025-07-10T04:56:44.898653+00:00",
                    "profit": 457471.1871,
                    "return_pct": 319.72,
                    "net_transfers": 634172.0,
                },
            },
        },
    }


def _lane(
    pair: str,
    side: str = "LONG",
    *,
    status: str = "LIVE_READY",
    method: str = "TREND_CONTINUATION",
    order_type: str = "STOP-ENTRY",
    same_pair_add_type: str | None = None,
    thesis_state: str | None = None,
    tp_atr_pips: float | None = None,
    price_percentile: float = 0.2,
    h1_regime: str = "TREND_UP",
    h4_regime: str = "TREND_UP",
    sl_lint_status: str | None = None,
) -> dict:
    metadata = {
        "entry_price_percentile_24h": price_percentile,
        "h1_regime": h1_regime,
        "h4_regime": h4_regime,
        "forecast_direction": "UP" if side == "LONG" else "DOWN",
    }
    if same_pair_add_type is not None:
        metadata["same_pair_add_type"] = same_pair_add_type
    if thesis_state is not None:
        metadata["thesis_state"] = thesis_state
    if tp_atr_pips is not None:
        metadata["tp_atr_pips"] = tp_atr_pips
    if sl_lint_status is not None:
        metadata["sl_lint_status"] = sl_lint_status
    return {
        "lane_id": f"shape:{pair}:{side}:{method}",
        "status": status,
        "live_blockers": [] if status == "LIVE_READY" else ["forecast confidence below live floor"],
        "risk_issues": [],
        "strategy_issues": [],
        "live_strategy_issues": [],
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": order_type,
            "market_context": {"method": method, "session": "LONDON"},
            "metadata": metadata,
        },
    }


class DiscretionaryTradeShapeTest(unittest.TestCase):
    def test_usd_jpy_precedent_does_not_force_usd_jpy_only_trading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_history_payload()))
            intents = root / "intents.json"
            intents.write_text(json.dumps({"results": [_lane("EUR_USD", "LONG")]}))
            target = root / "target.json"
            target.write_text(json.dumps({"target_trades_per_day": 10}))

            summary = build_operator_precedent_audit(
                manual_history_path=manual,
                order_intents_path=intents,
                target_state_path=target,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
            )

            self.assertEqual(summary.status, "OPERATOR_PRECEDENT_PASS")
            payload = json.loads((root / "audit.json").read_text())
            self.assertEqual(payload["precedent"]["winning_shape"]["primary_pair"], "USD_JPY")
            runtime = payload["runtime_alignment"]
            self.assertEqual(runtime["legacy_pair_direction_session_aligned_live_ready_lanes"], 0)
            self.assertEqual(runtime["aligned_live_ready_lanes"], 1)
            self.assertEqual(runtime["aligned_lanes"][0]["pair"], "EUR_USD")
            engine = runtime["trade_shape_engine"]
            self.assertTrue(engine["contract"]["does_not_force_usd_jpy_only_trading"])
            self.assertEqual(
                engine["pair_summaries"]["EUR_USD"]["precedent_match"]["status"],
                "MATCH",
            )

    def test_same_trade_shape_logic_applies_to_required_pairs(self) -> None:
        pairs = ["EUR_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
        engine = evaluate_trade_shape_engine({"results": [_lane(pair, "LONG") for pair in pairs]})

        self.assertEqual(set(engine["candidate_pairs"]), set(pairs))
        for pair in pairs:
            summary = engine["pair_summaries"][pair]
            shape = summary["precedent_match"]
            self.assertEqual(shape["status"], "MATCH")
            full_shape = next(item for item in engine["pair_evaluations"] if item["pair"] == pair)["trade_shape"]
            self.assertEqual(full_shape["cleanest_pair_expression"], pair)
            self.assertIn(full_shape["location_24h"], {"LOWER", "MIDDLE", "UPPER", "UNKNOWN"})
            self.assertIn(full_shape["tape_state"], {"TREND", "RANGE", "SQUEEZE", "FADE", "ROTATION"})
            self.assertIn(full_shape["building_style"], {"SINGLE", "BOUNDED_ADVERSE_ADD", "WITH_MOVE_PYRAMID"})

    def test_bounded_adverse_add_requires_alive_thesis_and_bounded_risk_on_any_pair(self) -> None:
        engine = evaluate_trade_shape_engine(
            {
                "results": [
                    _lane("EUR_USD", "LONG", same_pair_add_type="AVERAGE_INTO_ADVERSE", tp_atr_pips=3.2),
                    _lane(
                        "EUR_JPY",
                        "LONG",
                        status="DRY_RUN_BLOCKED",
                        same_pair_add_type="AVERAGE_INTO_ADVERSE",
                        thesis_state="WOUNDED",
                        tp_atr_pips=3.2,
                    ),
                    _lane("GBP_JPY", "SHORT", same_pair_add_type="AVERAGE_INTO_ADVERSE"),
                ]
            }
        )

        by_pair = {item["pair"]: item for item in engine["pair_evaluations"]}
        self.assertTrue(by_pair["EUR_USD"]["tradable"])
        self.assertIn(
            "BOUNDED_ADVERSE_ADD_ALLOWED",
            {item["code"] for item in by_pair["EUR_USD"]["allowed_behaviors"]},
        )
        self.assertFalse(by_pair["EUR_JPY"]["tradable"])
        self.assertIn(
            "BOUNDED_ADVERSE_ADD_BLOCKED",
            {item["code"] for item in by_pair["EUR_JPY"]["blocked_behaviors"]},
        )
        self.assertFalse(by_pair["GBP_JPY"]["tradable"])
        self.assertIn(
            "BOUNDED_ADVERSE_ADD_BLOCKED",
            {item["code"] for item in by_pair["GBP_JPY"]["blocked_behaviors"]},
        )

    def test_with_move_pyramid_is_blocked_across_all_pairs(self) -> None:
        pairs = ["USD_JPY", "EUR_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
        engine = evaluate_trade_shape_engine(
            {"results": [_lane(pair, "LONG", same_pair_add_type="PYRAMID_WITH_MOVE") for pair in pairs]}
        )

        for item in engine["pair_evaluations"]:
            self.assertFalse(item["tradable"], item["pair"])
            self.assertEqual(item["trade_shape"]["building_style"], "WITH_MOVE_PYRAMID")
            self.assertIn(
                "WITH_MOVE_PYRAMID_BLOCKED",
                {blocked["code"] for blocked in item["blocked_behaviors"]},
            )
            self.assertIn("With-move pyramiding", item["exact_reason_if_not_tradable"])

    def test_pair_specific_overlays_adjust_but_do_not_replace_common_scoring(self) -> None:
        engine = evaluate_trade_shape_engine(
            {
                "results": [
                    _lane("GBP_JPY", "LONG", same_pair_add_type="PYRAMID_WITH_MOVE"),
                    _lane("AUD_USD", "LONG"),
                    _lane("EUR_USD", "LONG"),
                ]
            }
        )
        by_pair = {item["pair"]: item for item in engine["pair_evaluations"]}

        gbp_jpy = by_pair["GBP_JPY"]
        self.assertLess(gbp_jpy["overlay_score_delta"], 0)
        self.assertLessEqual(gbp_jpy["trade_shape_score"], gbp_jpy["core_score_before_overlays"])
        self.assertIn("WITH_MOVE_PYRAMID_BLOCKED", {item["code"] for item in gbp_jpy["blocked_behaviors"]})
        self.assertTrue(all(item["overlay_only"] for item in gbp_jpy["pair_specific_adjustments"]))
        self.assertIn("GBP_JPY_SPREAD_NOISE_PENALTY", {item["code"] for item in gbp_jpy["pair_specific_adjustments"]})

        aud_usd = by_pair["AUD_USD"]
        self.assertIn("AUD_USD_NO_EDGE_SIZE_CAP", {item["code"] for item in aud_usd["pair_specific_adjustments"]})
        self.assertNotEqual(aud_usd["trade_shape_score"], aud_usd["core_score_before_overlays"])

        eur_usd = by_pair["EUR_USD"]
        self.assertIn(
            "EUR_USD_DIRECT_USD_THEME_EXPRESSION",
            {item["code"] for item in eur_usd["pair_specific_adjustments"]},
        )
        self.assertGreater(eur_usd["trade_shape_score"], eur_usd["core_score_before_overlays"])


if __name__ == "__main__":
    unittest.main()

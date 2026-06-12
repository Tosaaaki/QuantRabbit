from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
                "TOKYO": {"trades": 103, "net": -21887.5, "win_rate": 0.476, "payoff": 0.9},
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


def _manual_context_payload() -> dict:
    return {
        "status": "MANUAL_MARKET_CONTEXT_PASS",
        "bounded_replay_profile": {
            "by_h1_alignment": [
                {"bucket": "AGAINST_H1_TREND", "net_jpy": 1000.0},
                {"bucket": "WITH_H1_TREND", "net_jpy": -500.0},
            ],
            "by_m5_alignment": [
                {"bucket": "AGAINST_M5_TREND", "net_jpy": 1200.0},
                {"bucket": "WITH_M5_TREND", "net_jpy": -300.0},
            ],
            "by_side_h1_alignment": [
                {"bucket": "LONG_AGAINST_H1_TREND", "net_jpy": 900.0},
                {"bucket": "LONG_WITH_H1_TREND", "net_jpy": -600.0},
            ],
            "by_side_entry_location_24h": [
                {"bucket": "LONG_LOWER_THIRD_24H", "net_jpy": 800.0},
                {"bucket": "LONG_UPPER_THIRD_24H", "net_jpy": -700.0},
            ],
            "by_session_jst": [
                {"bucket": "LONDON_AM", "net_jpy": 600.0},
                {"bucket": "NY_OVERLAP", "net_jpy": -400.0},
            ],
        },
    }


def _live_ready_intents(
    *,
    h1_regime: str,
    m5_regime: str,
    entry_price_percentile_24h: float,
    session: str = "LONDON",
) -> dict:
    return {
        "results": [
            {
                "lane_id": "trend_trader:USD_JPY:LONG:TREND_CONTINUATION",
                "status": "LIVE_READY",
                "intent": {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "order_type": "STOP-ENTRY",
                    "market_context": {
                        "method": "TREND_CONTINUATION",
                        "session": session,
                    },
                    "metadata": {
                        "h1_regime": h1_regime,
                        "m5_regime": m5_regime,
                        "entry_price_percentile_24h": entry_price_percentile_24h,
                    },
                },
            }
        ]
    }


class OperatorPrecedentAuditTest(unittest.TestCase):
    def test_verifies_manual_precedent_and_current_aligned_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_history_payload()))
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "trend_trader:USD_JPY:LONG:TREND_CONTINUATION",
                                "status": "LIVE_READY",
                                "intent": {
                                    "pair": "USD_JPY",
                                    "side": "LONG",
                                    "order_type": "STOP-ENTRY",
                                    "market_context": {
                                        "method": "TREND_CONTINUATION",
                                        "session": "LONDON",
                                    },
                                    "metadata": {},
                                },
                            }
                        ]
                    }
                )
            )
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
            self.assertEqual(summary.best_30d_return_pct, 319.72)
            self.assertEqual(summary.live_ready_lanes, 1)
            self.assertEqual(summary.aligned_live_ready_lanes, 1)
            payload = json.loads((root / "audit.json").read_text())
            self.assertTrue(payload["operator_claim"]["verified"])
            self.assertEqual(payload["precedent"]["winning_shape"]["primary_sessions"], ["LONDON_AM", "NY_OVERLAP"])
            self.assertTrue(payload["contract"]["advisory_only"])
            self.assertIn("RiskEngine", payload["contract"]["cannot_override"])

    def test_marks_precedent_aligned_lane_conflicting_when_manual_technical_context_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_history_payload()))
            manual_context = root / "manual_context.json"
            manual_context.write_text(json.dumps(_manual_context_payload()))
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    _live_ready_intents(
                        h1_regime="TREND_UP",
                        m5_regime="TREND_UP",
                        entry_price_percentile_24h=0.82,
                    )
                )
            )
            target = root / "target.json"
            target.write_text(json.dumps({"target_trades_per_day": 10}))

            summary = build_operator_precedent_audit(
                manual_history_path=manual,
                manual_context_path=manual_context,
                order_intents_path=intents,
                target_state_path=target,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
            )

            self.assertEqual(summary.status, "OPERATOR_PRECEDENT_WARN")
            payload = json.loads((root / "audit.json").read_text())
            alignment = payload["runtime_alignment"]["manual_context_alignment"]
            self.assertEqual(alignment["conflicting_aligned_lanes"], 1)
            self.assertEqual(
                alignment["conflicting_lanes"][0]["lane_id"],
                "trend_trader:USD_JPY:LONG:TREND_CONTINUATION",
            )
            self.assertEqual(alignment["conflicting_lanes"][0]["h1_alignment"], "WITH_H1_TREND")
            self.assertIn(
                "by_side_entry_location_24h:LONG_UPPER_THIRD_24H",
                alignment["conflicting_lanes"][0]["conflicting_buckets"],
            )
            self.assertIn(
                "by_side_h1_alignment:LONG_WITH_H1_TREND",
                alignment["conflicting_lanes"][0]["conflicting_buckets"],
            )
            check_names = {item["check_name"] for item in payload["checks"]}
            self.assertIn("manual_context_alignment_conflict", check_names)

    def test_marks_precedent_aligned_lane_conflicting_when_manual_session_context_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_history_payload()))
            manual_context = root / "manual_context.json"
            manual_context.write_text(json.dumps(_manual_context_payload()))
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    _live_ready_intents(
                        h1_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                        entry_price_percentile_24h=0.12,
                        session="NY_OVERLAP",
                    )
                )
            )
            target = root / "target.json"
            target.write_text(json.dumps({"target_trades_per_day": 10}))

            summary = build_operator_precedent_audit(
                manual_history_path=manual,
                manual_context_path=manual_context,
                order_intents_path=intents,
                target_state_path=target,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
            )

            self.assertEqual(summary.status, "OPERATOR_PRECEDENT_WARN")
            payload = json.loads((root / "audit.json").read_text())
            alignment = payload["runtime_alignment"]["manual_context_alignment"]
            self.assertEqual(alignment["conflicting_aligned_lanes"], 1)
            self.assertIn(
                "by_session_jst:NY_OVERLAP",
                alignment["conflicting_lanes"][0]["conflicting_buckets"],
            )

    def test_marks_precedent_aligned_lane_compatible_when_manual_technical_context_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_history_payload()))
            manual_context = root / "manual_context.json"
            manual_context.write_text(json.dumps(_manual_context_payload()))
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    _live_ready_intents(
                        h1_regime="TREND_DOWN",
                        m5_regime="TREND_DOWN",
                        entry_price_percentile_24h=0.12,
                    )
                )
            )
            target = root / "target.json"
            target.write_text(json.dumps({"target_trades_per_day": 10}))

            summary = build_operator_precedent_audit(
                manual_history_path=manual,
                manual_context_path=manual_context,
                order_intents_path=intents,
                target_state_path=target,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
            )

            self.assertEqual(summary.status, "OPERATOR_PRECEDENT_PASS")
            payload = json.loads((root / "audit.json").read_text())
            alignment = payload["runtime_alignment"]["manual_context_alignment"]
            self.assertEqual(alignment["conflicting_aligned_lanes"], 0)
            self.assertEqual(
                alignment["compatible_lanes"][0]["lane_id"],
                "trend_trader:USD_JPY:LONG:TREND_CONTINUATION",
            )
            self.assertIn(
                "by_h1_alignment:AGAINST_H1_TREND",
                alignment["compatible_lanes"][0]["supporting_buckets"],
            )
            self.assertIn(
                "by_side_h1_alignment:LONG_AGAINST_H1_TREND",
                alignment["compatible_lanes"][0]["supporting_buckets"],
            )
            self.assertIn(
                "by_session_jst:LONDON_AM",
                alignment["compatible_lanes"][0]["supporting_buckets"],
            )

    def test_uses_trade_close_window_with_oanda_nanosecond_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _manual_history_payload()
            payload["exit_events"] = 2
            payload["closed_trades"] = 2
            payload["reduced_trades"] = 0
            payload["trades"] = [
                {"close_time": "2025-06-13T00:00:00.123456789Z"},
                {"close_time": "2025-06-15T00:00:00.987654321Z"},
            ]
            manual = root / "manual.json"
            manual.write_text(json.dumps(payload))
            intents = root / "intents.json"
            intents.write_text(json.dumps({"results": []}))
            target = root / "target.json"
            target.write_text(json.dumps({}))

            build_operator_precedent_audit(
                manual_history_path=manual,
                order_intents_path=intents,
                target_state_path=target,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
            )

            audit = json.loads((root / "audit.json").read_text())
            active_window = audit["precedent"]["sample"]["active_window"]
            self.assertEqual(active_window["calendar_days"], 2.0)
            self.assertEqual(active_window["exit_events_per_calendar_day"], 1.0)

    def test_missing_manual_history_is_audited_as_blocked_not_silent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            intents.write_text(json.dumps({"results": []}))
            target = root / "target.json"
            target.write_text(json.dumps({}))

            summary = build_operator_precedent_audit(
                manual_history_path=root / "missing.json",
                order_intents_path=intents,
                target_state_path=target,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
            )

            self.assertEqual(summary.status, "OPERATOR_PRECEDENT_BLOCKED")
            self.assertGreater(summary.blockers, 0)
            payload = json.loads((root / "audit.json").read_text())
            self.assertIn("manual history artifact missing", payload["blockers"][0])


if __name__ == "__main__":
    unittest.main()

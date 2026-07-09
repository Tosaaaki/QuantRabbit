from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.forecast_pattern_refresh import ForecastPatternRefresh


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


class ForecastPatternRefreshTest(unittest.TestCase):
    def test_usdcad_range_forecast_becomes_range_rail_geometry_repair_queue(self) -> None:
        now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _paths(root)
            top_lane = {
                "lane_id": lane_id,
                "pair": "USD_CAD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "LIMIT",
                "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                "blockers": [
                    "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                    "SPREAD_TOO_WIDE",
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
                "forecast_audit": {
                    "status": "FORECAST_PATTERN_REFRESH_REQUIRED",
                    "latest": {
                        "timestamp_utc": "2026-07-09T10:58:00+00:00",
                        "direction": "RANGE",
                        "confidence": 1.0,
                        "current_price": 1.4181,
                        "range_low_price": 1.41528,
                        "range_high_price": 1.41899,
                    },
                },
                "tp_proof_audit": {
                    "status": "TP_PROOF_COLLECTION_REQUIRED",
                    "tp_proof_count": 1,
                    "tp_proof_floor": 20,
                    "remaining_samples": 19,
                },
            }
            _write_json(
                paths["entry_recovery"],
                {
                    "schema_version": "entry_frequency_recovery_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": top_lane,
                    "forecast_pattern_tuning_queue": [
                        {
                            "priority": 1,
                            "lane_id": lane_id,
                            "action_type": "FORECAST_PATTERN_REFRESH",
                            "description": "Retune USD_CAD LONG around RANGE rail geometry.",
                            "preserve_blockers": [
                                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                                "SPREAD_TOO_WIDE",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            ],
                        }
                    ],
                    "next_contract_prompt": "Consume data/entry_frequency_recovery.json; do not send.",
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "current_state": {"active_opportunity_board": {"top_lane": top_lane}},
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["active_board"],
                {
                    "top_lane": top_lane,
                    "ranked_active_lanes": [top_lane],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent(
                            lane_id,
                            "LIMIT",
                            [
                                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                                "SPREAD_TOO_WIDE",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            ],
                        ),
                        _intent(
                            "range_trader:USD_CAD:LONG:RANGE_ROTATION",
                            "LIMIT",
                            [
                                "SPREAD_TOO_WIDE",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                                "EXHAUSTION_RANGE_CHASE",
                            ],
                        ),
                        _intent(
                            "range_trader:USD_CAD:LONG:RANGE_ROTATION:MARKET",
                            "MARKET",
                            [
                                "RANGE_MARKET_NOT_AT_RAIL",
                                "SPREAD_TOO_WIDE",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            ],
                        ),
                    ],
                },
            )
            _write_jsonl(
                paths["forecast_history"],
                [
                    {
                        "timestamp_utc": "2026-07-09T10:58:00+00:00",
                        "pair": "USD_CAD",
                        "direction": "RANGE",
                        "confidence": 1.0,
                        "current_price": 1.4181,
                        "range_low_price": 1.41528,
                        "range_high_price": 1.41899,
                    }
                ],
            )
            _write_jsonl(
                paths["projection_ledger"],
                [
                    {
                        "timestamp_emitted_utc": "2026-07-09T10:58:00+00:00",
                        "pair": "USD_CAD",
                        "signal_name": "directional_forecast",
                        "direction": "RANGE",
                        "confidence": 1.0,
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                    },
                    {
                        "timestamp_emitted_utc": "2026-07-09T10:58:00+00:00",
                        "pair": "USD_CAD",
                        "signal_name": "liquidity_sweep_low",
                        "direction": "UP",
                        "confidence": 0.99,
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                    },
                ],
            )

            summary = ForecastPatternRefresh(
                entry_frequency_recovery_path=paths["entry_recovery"],
                active_trader_contract_path=paths["active_contract"],
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        self.assertEqual(summary.status, "FORECAST_PATTERN_REFRESH_BUILT")
        self.assertEqual(summary.target_lane_id, lane_id)
        self.assertFalse(summary.live_permission_allowed)
        self.assertEqual(payload["live_side_effects"], [])
        self.assertFalse(payload["live_permission_allowed"])
        top = payload["top_lane"]
        self.assertEqual(top["forecast_range_box"]["status"], "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL")
        self.assertEqual(
            top["range_rotation_counterpart"]["status"],
            "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL",
        )
        self.assertTrue(top["blocker_preservation"]["spread_too_wide_not_ignored"])
        self.assertTrue(top["blocker_preservation"]["negative_expectancy_visible"])
        action_types = [row["action_type"] for row in payload["next_actions"]]
        self.assertIn("RANGE_RAIL_GEOMETRY_REPAIR", action_types)
        self.assertIn("VERIFY_TRIGGER_PROJECTIONS", action_types)
        self.assertIn("EXACT_TP_PROOF_COLLECTION", action_types)
        self.assertIn("Consume data/forecast_pattern_refresh.json", payload["next_contract_prompt"])
        self.assertIn("RANGE_RAIL_GEOMETRY_REPAIR", payload["next_contract_prompt"])
        self.assertIn("RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL", report)

    def test_no_current_target_is_read_only_noop(self) -> None:
        now = datetime(2026, 7, 9, 12, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            summary = ForecastPatternRefresh(
                entry_frequency_recovery_path=paths["entry_recovery"],
                active_trader_contract_path=paths["active_contract"],
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.status, "NO_FORECAST_PATTERN_REFRESH_TARGET")
        self.assertIsNone(summary.target_lane_id)
        self.assertEqual(payload["top_lane"], {})
        self.assertEqual(payload["next_actions"], [])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])


def _intent(lane_id: str, order_type: str, blocker_codes: list[str]) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "status": "DRY_RUN_BLOCKED",
        "risk_allowed": False,
        "live_blocker_codes": blocker_codes,
        "risk_issues": [{"severity": "BLOCK", "code": code, "message": code} for code in blocker_codes],
        "strategy_issues": [],
        "intent": {
            "order_type": order_type,
            "units": 1000,
            "entry_price": 1.418,
            "take_profit_price": 1.4195,
            "stop_loss_price": 1.416,
        },
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "entry_recovery": root / "data" / "entry_frequency_recovery.json",
        "active_contract": root / "data" / "active_trader_contract.json",
        "active_board": root / "data" / "active_opportunity_board.json",
        "order_intents": root / "data" / "order_intents.json",
        "forecast_history": root / "data" / "forecast_history.jsonl",
        "projection_ledger": root / "data" / "projection_ledger.jsonl",
        "output": root / "data" / "forecast_pattern_refresh.json",
        "report": root / "docs" / "forecast_pattern_refresh_report.md",
    }


if __name__ == "__main__":
    unittest.main()

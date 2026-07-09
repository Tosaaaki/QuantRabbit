from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.range_rail_geometry_repair import RangeRailGeometryRepair


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class RangeRailGeometryRepairTest(unittest.TestCase):
    def test_usdcad_long_not_at_discount_rail_builds_recheck_condition(self) -> None:
        now = datetime(2026, 7, 9, 12, 45, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(paths["forecast"], _forecast_payload(now, lane_id, current=1.4181))
            _write_json(paths["active_board"], {"top_lane": {"lane_id": lane_id, "pair": "USD_CAD", "direction": "LONG", "strategy_family": "BREAKOUT_FAILURE"}})
            _write_json(paths["order_intents"], {"results": [_range_intent(entry=None, tp=None, sl=None)]})

            summary = RangeRailGeometryRepair(
                forecast_pattern_refresh_path=paths["forecast"],
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        self.assertEqual(summary.status, "RANGE_RAIL_RECHECK_BUILT")
        self.assertEqual(summary.target_lane_id, lane_id)
        self.assertFalse(summary.live_permission_allowed)
        self.assertEqual(payload["live_side_effects"], [])
        self.assertFalse(payload["live_permission_allowed"])
        top = payload["top_lane"]
        self.assertEqual(top["range_box"]["rail_status"], "RANGE_RAIL_NOT_REACHED")
        self.assertAlmostEqual(top["range_box"]["box_position"], 0.7601)
        self.assertEqual(top["rail_success_condition"]["required_box_position_lte"], 0.35)
        self.assertEqual(payload["next_actions"][0]["action_type"], "WAIT_FOR_RANGE_RAIL_RECHECK")
        self.assertIn("VERIFY_TRIGGER_PROJECTIONS", [row["action_type"] for row in payload["next_actions"]])
        self.assertIn("EXACT_TP_PROOF_COLLECTION", [row["action_type"] for row in payload["next_actions"]])
        self.assertTrue(top["blocker_preservation"]["spread_too_wide_not_ignored"])
        self.assertTrue(top["blocker_preservation"]["negative_expectancy_visible"])
        self.assertTrue(top["blocker_preservation"]["range_location_blockers_not_ignored"])
        self.assertIn("WAIT_FOR_RANGE_RAIL_RECHECK", payload["next_contract_prompt"])
        self.assertIn("RANGE_RAIL_NOT_REACHED", report)

    def test_long_discount_rail_with_limit_geometry_stays_proof_blocked(self) -> None:
        now = datetime(2026, 7, 9, 12, 50, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(paths["forecast"], _forecast_payload(now, lane_id, current=1.4155))
            _write_json(paths["active_board"], {"top_lane": {"lane_id": lane_id, "pair": "USD_CAD", "direction": "LONG", "strategy_family": "BREAKOUT_FAILURE"}})
            _write_json(
                paths["order_intents"],
                {"results": [_range_intent(entry=1.41555, tp=1.4168, sl=1.4149)]},
            )

            summary = RangeRailGeometryRepair(
                forecast_pattern_refresh_path=paths["forecast"],
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.status, "RANGE_RAIL_GEOMETRY_READY_PROOF_BLOCKED")
        top = payload["top_lane"]
        self.assertEqual(top["range_box"]["rail_status"], "RANGE_RAIL_REACHED")
        self.assertEqual(top["counterpart_geometry"]["status"], "COUNTERPART_GEOMETRY_READY")
        self.assertTrue(top["counterpart_geometry"]["entry_inside_box"])
        self.assertTrue(top["counterpart_geometry"]["tp_inside_box"])
        self.assertTrue(top["counterpart_geometry"]["sl_outside_box"])
        self.assertEqual(payload["next_actions"][0]["action_type"], "RANGE_ROTATION_GEOMETRY_READY_PROOF_BLOCKED")
        self.assertIn("Geometry is already ready", payload["next_contract_prompt"])
        self.assertIn("do not repeat range-rail geometry repair", payload["next_contract_prompt"])
        self.assertIn("VERIFY_TRIGGER_PROJECTIONS", payload["next_contract_prompt"])
        self.assertIn("EXACT_TP_PROOF_COLLECTION", payload["next_contract_prompt"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_live_intent_entry_tp_sl_aliases_feed_counterpart_geometry(self) -> None:
        now = datetime(2026, 7, 9, 12, 55, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(paths["forecast"], _forecast_payload(now, lane_id, current=1.4155))
            _write_json(paths["active_board"], {"top_lane": {"lane_id": lane_id, "pair": "USD_CAD", "direction": "LONG", "strategy_family": "BREAKOUT_FAILURE"}})
            _write_json(
                paths["order_intents"],
                {
                    "results": [
                        _range_intent(
                            entry=1.41555,
                            tp=1.4168,
                            sl=1.4149,
                            intent_price_aliases=True,
                        )
                    ]
                },
            )

            RangeRailGeometryRepair(
                forecast_pattern_refresh_path=paths["forecast"],
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        geometry = payload["top_lane"]["counterpart_geometry"]
        self.assertEqual(payload["status"], "RANGE_RAIL_GEOMETRY_READY_PROOF_BLOCKED")
        self.assertEqual(geometry["entry_price"], 1.41555)
        self.assertEqual(geometry["take_profit_price"], 1.4168)
        self.assertEqual(geometry["stop_loss_price"], 1.4149)
        self.assertEqual(geometry["status"], "COUNTERPART_GEOMETRY_READY")
        self.assertEqual(payload["next_actions"][0]["action_type"], "RANGE_ROTATION_GEOMETRY_READY_PROOF_BLOCKED")
        self.assertIn("do not repeat range-rail geometry repair", payload["next_contract_prompt"])
        self.assertIn("EXACT_TP_PROOF_COLLECTION", payload["next_contract_prompt"])


def _forecast_payload(now: datetime, lane_id: str, *, current: float) -> dict[str, Any]:
    return {
        "schema_version": "forecast_pattern_refresh_v1",
        "generated_at_utc": now.isoformat(),
        "status": "FORECAST_PATTERN_REFRESH_BUILT",
        "read_only": True,
        "live_side_effects": [],
        "live_permission_allowed": False,
        "top_lane": {
            "lane_id": lane_id,
            "pair": "USD_CAD",
            "direction": "LONG",
            "strategy_family": "BREAKOUT_FAILURE",
            "vehicle": "LIMIT",
            "status": "EVIDENCE_ACQUISITION",
            "blockers": [
                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                "SPREAD_TOO_WIDE",
                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
            ],
            "forecast_range_box": {
                "status": "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL",
                "box_position": round((current - 1.41528) / (1.41899 - 1.41528), 4),
                "latest": {
                    "timestamp_utc": "2026-07-09T11:13:09.833095+00:00",
                    "direction": "RANGE",
                    "confidence": 1.0,
                    "current_price": current,
                    "range_low_price": 1.41528,
                    "range_high_price": 1.41899,
                },
            },
            "range_rotation_counterpart": {
                "status": "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL",
                "preferred_lane_id": "range_trader:USD_CAD:LONG:RANGE_ROTATION",
                "blocker_codes": [
                    "SPREAD_TOO_WIDE",
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                    "EXHAUSTION_RANGE_CHASE",
                ],
            },
        },
        "next_actions": [
            {
                "priority": 1,
                "lane_id": lane_id,
                "action_type": "RANGE_RAIL_GEOMETRY_REPAIR",
                "description": "Repair range rail geometry.",
                "preserve_blockers": ["RANGE_FORECAST_REQUIRES_RANGE_ROTATION"],
            }
        ],
    }


def _range_intent(
    *,
    entry: float | None,
    tp: float | None,
    sl: float | None,
    intent_price_aliases: bool = False,
) -> dict[str, Any]:
    blockers = [
        "SPREAD_TOO_WIDE",
        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
        "RANGE_ROTATION_BROADER_LOCATION_CHASE",
        "EXHAUSTION_RANGE_CHASE",
    ]
    intent = {
        "order_type": "LIMIT",
        "units": 1000,
    }
    if intent_price_aliases:
        intent.update({"entry": entry, "tp": tp, "sl": sl})
    else:
        intent.update(
            {
                "entry_price": entry,
                "take_profit_price": tp,
                "stop_loss_price": sl,
            }
        )
    return {
        "lane_id": "range_trader:USD_CAD:LONG:RANGE_ROTATION",
        "status": "DRY_RUN_BLOCKED",
        "risk_allowed": False,
        "live_blocker_codes": blockers,
        "risk_issues": [{"severity": "BLOCK", "code": code, "message": code} for code in blockers],
        "strategy_issues": [],
        "intent": intent,
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "forecast": root / "data" / "forecast_pattern_refresh.json",
        "active_board": root / "data" / "active_opportunity_board.json",
        "order_intents": root / "data" / "order_intents.json",
        "output": root / "data" / "range_rail_geometry_repair.json",
        "report": root / "docs" / "range_rail_geometry_repair_report.md",
    }


if __name__ == "__main__":
    unittest.main()

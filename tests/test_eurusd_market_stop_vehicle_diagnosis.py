from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_eurusd_market_stop_vehicle_diagnosis as diagnosis


class EurusdMarketStopVehicleDiagnosisTest(unittest.TestCase):
    def test_market_and_stop_are_split_without_limit_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            _write_json(
                data / "eurusd_short_breakout_failure_vehicle_split_diagnosis.json",
                {
                    "vehicle_split": {
                        "limit_order_summary": {
                            "samples": 4,
                            "wins": 4,
                            "losses": 0,
                            "net_jpy": 3255.0938,
                        },
                        "limit_order_samples": [
                            _sample("472732", "LIMIT_ORDER", 813.7706, entry_order_price=1.14486)
                        ],
                        "market_order_samples": [
                            _sample("470705", "MARKET_ORDER", 677.4009, entry_order_price=None),
                            _sample("470788", "MARKET_ORDER", 256.5150, entry_order_price=None),
                        ],
                        "stop_order_samples": [
                            _sample("471292", "STOP_ORDER", 4572.2661, entry_order_price=1.16077),
                            _sample("471306", "STOP_ORDER", 784.1109, entry_order_price=1.15960),
                        ],
                    }
                },
            )
            _write_json(
                data / "eurusd_short_breakout_failure_limit_sample_mining.json",
                {
                    "status": "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED_STILL_UNDERSAMPLED",
                    "sample_floor": {
                        "required_exact_limit_samples": 20,
                        "current_replayed_exact_limit_samples": 4,
                        "additional_acceptable_local_samples_found": 0,
                        "remaining_exact_limit_samples": 16,
                        "floor_met": False,
                    },
                },
            )
            for name in [
                "active_trader_contract",
                "eurusd_short_breakout_failure_spread_slippage_proof",
                "eurusd_short_breakout_failure_proof_floor_update",
                "harvest_live_grade_path",
                "operator_review_report",
                "trader_goal_loop_orchestrator",
            ]:
                _write_json(data / f"{name}.json", {"status": "OK", "remaining_blockers": []})

            with patch.object(diagnosis, "ROOT", root):
                payload = diagnosis.build_payload("2026-07-08T00:00:00Z")

        self.assertEqual(payload["status"], "MARKET_STOP_VEHICLE_PROMISING_STILL_BLOCKED")
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertEqual(payload["market_harvest_vehicle"]["sample_summary"]["sample_count"], 2)
        self.assertEqual(payload["stop_harvest_vehicle"]["sample_summary"]["sample_count"], 2)
        self.assertFalse(payload["market_harvest_vehicle"]["proof_boundary"]["included_in_limit_proof"])
        self.assertFalse(payload["stop_harvest_vehicle"]["proof_boundary"]["included_in_limit_proof"])
        self.assertFalse(payload["safety_checks"]["limit_proof_mixes_market_stop_samples"])
        self.assertEqual(
            payload["vehicle_comparison"]["market_vs_stop"]["closer_to_4x_active_path"],
            "STOP_HARVEST",
        )
        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertIn("MARKET_HARVEST_EXACT_REPLAY_REQUIRED", blocker_codes)
        self.assertIn("STOP_HARVEST_EXACT_REPLAY_REQUIRED", blocker_codes)
        self.assertIn("MARKET_STOP_NOT_ALLOWED_IN_LIMIT_PROOF", blocker_codes)


def _sample(trade_id: str, order_type: str, pl: float, *, entry_order_price: float | None) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "entry_order_type": order_type,
        "entry_price": entry_order_price or 1.16000,
        "entry_order_price": entry_order_price,
        "tp_order_price": 1.15900,
        "entry_fill_vs_order_price_pips": 0.0 if entry_order_price is not None else None,
        "entry_spread_pips": 0.8,
        "exit_spread_pips": 0.8,
        "realized_pl_jpy": pl,
        "observed_cost_inclusive_pass": True,
        "market_close_mixed_in": False,
        "bidask_replay_status": "MISSING_EXACT_INDEPENDENT_S5_BA_REPLAY",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

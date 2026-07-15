from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.forecast_learning import (
    forecast_orientation_decision,
    scored_row_feature_values,
    train_forecast_orientation_model,
    verify_forecast_orientation_model,
)


class ForecastLearningTest(unittest.TestCase):
    def test_chronological_bidask_model_learns_keep_or_invert_without_no_trade(self) -> None:
        direct: list[dict[str, object]] = []
        inverse: list[dict[str, object]] = []
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(120):
            direction = "UP" if index % 2 == 0 else "DOWN"
            timestamp = started + timedelta(hours=index * 2)
            common = {
                "source_index": index,
                "timestamp_utc": timestamp.isoformat(),
                "pair": "USD_JPY",
                "direction": direction,
                "confidence_bucket": "0.65-0.75",
                "raw_confidence_bucket": "0.65-0.75",
                "score_margin_bucket": "10-20",
                "range_competition": "DIRECTIONAL_MARGIN_DOMINATES",
                "utc_session_bucket": "UTC_00_08",
                "primary_driver_family": "MOMENTUM",
                "primary_against_driver_family": "OTHER",
                "technical_regime": "TREND_WEAK",
                "technical_atr_band": "NORMAL",
                "technical_spread_band": "NORMAL",
                "technical_range_location_24h": "MIDDLE",
                "technical_structure_alignment": "ALIGNED",
                "technical_situation": "ASIA_TREND",
                "technical_selected_method": "TREND_CONTINUATION",
                "technical_family_direction_alignment": "ALIGNED",
                "horizon_min": 60,
            }
            direct_pips = 3.0 if direction == "UP" else -3.0
            inverse_pips = -3.0 if direction == "UP" else 3.0
            direct.append({**common, "final_pips": direct_pips})
            inverse.append(
                {
                    **common,
                    "direction": "DOWN" if direction == "UP" else "UP",
                    "final_pips": inverse_pips,
                }
            )

        model = train_forecast_orientation_model(
            direct,
            inverse,
            source="unit",
            provenance={"forecast_rows_sha256": "a" * 64},
        )

        self.assertTrue(verify_forecast_orientation_model(model))
        self.assertEqual(model["status"], "ENABLED")
        self.assertEqual(model["technical_feature_training_status"], "AVAILABLE")
        self.assertEqual(model["training_feature_coverage"]["technical_regime"], 1.0)
        self.assertEqual(
            model["training_provenance"]["forecast_rows_sha256"],
            "a" * 64,
        )
        self.assertGreater(
            model["validation_metrics"]["selected_avg_final_pips"],
            0.0,
        )
        down_features = scored_row_feature_values(direct[1])
        decision = forecast_orientation_decision(
            model,
            original_direction="DOWN",
            features=down_features,
        )
        self.assertEqual(decision["orientation"], "INVERSE")
        self.assertEqual(decision["direction"], "UP")
        self.assertTrue(decision["ordinary_correction_applied"])

    def test_model_digest_tamper_fails_closed_to_direct_prediction(self) -> None:
        direct = []
        inverse = []
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(60):
            row = {
                "source_index": index,
                "timestamp_utc": (started + timedelta(hours=index * 2)).isoformat(),
                "pair": "EUR_USD",
                "direction": "UP",
                "horizon_min": 60,
                "final_pips": 1.0,
            }
            direct.append(row)
            inverse.append({**row, "direction": "DOWN", "final_pips": -1.0})
        model = train_forecast_orientation_model(direct, inverse, source="unit")
        if model.get("model_sha256") is None:
            self.skipTest("synthetic split did not produce a model")
        tampered = copy.deepcopy(model)
        tampered["model"]["intercept"] = 99.0

        self.assertFalse(verify_forecast_orientation_model(tampered))
        decision = forecast_orientation_decision(
            tampered,
            original_direction="UP",
            features=scored_row_feature_values(direct[0]),
        )
        self.assertEqual(decision["direction"], "UP")
        self.assertFalse(decision["ordinary_correction_applied"])

    def test_model_discloses_missing_legacy_technical_feature_coverage(self) -> None:
        direct = []
        inverse = []
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(90):
            direct_pips = 1.0 if index % 2 == 0 else -1.0
            row = {
                "source_index": index,
                "timestamp_utc": (started + timedelta(hours=index * 2)).isoformat(),
                "pair": "EUR_USD",
                "direction": "UP",
                "horizon_min": 60,
                "final_pips": direct_pips,
            }
            direct.append(row)
            inverse.append(
                {**row, "direction": "DOWN", "final_pips": -direct_pips}
            )

        model = train_forecast_orientation_model(direct, inverse, source="unit")

        self.assertTrue(verify_forecast_orientation_model(model))
        self.assertEqual(
            model["technical_feature_training_status"],
            "MISSING_IN_LEGACY_FORECAST_HISTORY",
        )
        self.assertEqual(model["training_feature_coverage"]["technical_regime"], 0.0)


if __name__ == "__main__":
    unittest.main()

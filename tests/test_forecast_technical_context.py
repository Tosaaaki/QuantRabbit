from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from quant_rabbit.strategy.forecast_persistence_tracker import record_forecast
from quant_rabbit.strategy.forecast_technical_context import (
    build_forecast_technical_context,
    technical_context_sha256,
    verify_forecast_technical_context,
)


def _chart() -> dict:
    return {
        "confluence": {
            "dominant_regime": "TREND_UP",
            "atr_percentile_24h": 0.82,
            "price_percentile_24h": 0.88,
            "price_percentile_7d": 0.42,
        },
        "views": [
            {
                "granularity": "M5",
                "regime_reading": {"state": "TREND_STRONG", "atr_percentile": 80.0},
                "indicators": {"atr_pips": 2.0},
                "structure": {
                    "structure_events": [
                        {"kind": "CHOCH_DOWN", "index": 8, "close_confirmed": True},
                        {"kind": "BOS_UP", "index": 10, "close_confirmed": True},
                    ]
                },
            },
            {
                "granularity": "M15",
                "regime_reading": {"state": "TREND_WEAK", "atr_percentile": 60.0},
                "indicators": {"atr_pips": 5.0},
                "structure": {
                    "structure_events": [
                        {
                            "kind": "CHOCH_DOWN",
                            "index": 15,
                            "close_confirmed": True,
                            "timestamp": "2026-07-13T01:00:00Z",
                        }
                    ]
                },
            },
            {
                "granularity": "H1",
                "regime_reading": {"state": "RANGE", "atr_percentile": 20.0},
                "indicators": {"atr_pips": 14.0},
                "structure": {
                    "structure_events": [
                        {"kind": "BOS_UP", "index": 20, "close_confirmed": False},
                        {"kind": "BOS_DOWN", "index": 18, "close_confirmed": True},
                    ]
                },
            },
        ],
    }


class ForecastTechnicalContextTest(unittest.TestCase):
    def test_builds_normalized_content_addressed_context(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.123456789,
            spread_pips=0.8,
        )

        self.assertEqual(context["schema_version"], "technical_context_v1")
        self.assertEqual(context["regime"]["primary"], "TREND_WEAK")
        self.assertEqual(context["volatility"]["primary_atr_band"], "NORMAL")
        self.assertEqual(context["execution"]["spread_band"], "NORMAL")
        self.assertEqual(context["location"]["range_location_24h"], "UPPER")
        self.assertEqual(context["structure"]["primary_timeframe"], "H1")
        self.assertEqual(context["structure"]["primary_direction"], "DOWN")
        self.assertEqual(context["identity"]["pair"], "EUR_USD")
        self.assertIn("trend_score", context["families"]["by_timeframe"]["M15"])
        self.assertTrue(context["completeness"]["complete"])
        self.assertEqual(context["context_sha256"], technical_context_sha256(context))
        self.assertEqual(verify_forecast_technical_context(context), (True, None))
        self.assertEqual(
            verify_forecast_technical_context(
                context,
                pair="EUR_USD",
                current_price=1.12346,
            ),
            (True, None),
        )
        self.assertEqual(
            verify_forecast_technical_context(
                context,
                pair="EUR_USD",
                current_price=1.12360,
            ),
            (False, "TECHNICAL_CONTEXT_PRICE_MISMATCH"),
        )

        other_pair = build_forecast_technical_context(
            _chart(),
            pair="GBP_USD",
            current_price=1.123456789,
            spread_pips=0.8,
        )
        self.assertNotEqual(context["context_sha256"], other_pair["context_sha256"])
        self.assertEqual(
            verify_forecast_technical_context(context, pair="GBP_USD"),
            (False, "TECHNICAL_CONTEXT_PAIR_MISMATCH"),
        )

    def test_hash_detects_point_in_time_context_tampering(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        tampered = copy.deepcopy(context)
        tampered["location"]["range_location_24h"] = "LOWER"

        self.assertEqual(
            verify_forecast_technical_context(tampered),
            (False, "TECHNICAL_CONTEXT_HASH_MISMATCH"),
        )

        expanded = copy.deepcopy(context)
        expanded["raw_candles"] = [1, 2, 3]
        expanded["context_sha256"] = technical_context_sha256(expanded)
        self.assertEqual(
            verify_forecast_technical_context(expanded),
            (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
        )

    def test_record_forecast_persists_exact_context_and_rejects_bad_hash(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        forecast = SimpleNamespace(
            pair="EUR_USD",
            direction="UP",
            confidence=0.6,
            current_price=1.1,
            invalidation_price=1.09,
            target_price=1.12,
            horizon_min=60,
            drivers_for=(),
            drivers_against=(),
            rationale_summary="test",
            technical_context_v1=context,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(record_forecast(forecast, data_root=root, cycle_id="cycle-1"))
            row = json.loads((root / "forecast_history.jsonl").read_text())
            self.assertEqual(row["technical_context_v1"], context)
            receipt = json.loads((root / "forecast_emission_receipts.jsonl").read_text())
            self.assertEqual(receipt["schema_version"], "QR_FORECAST_EMISSION_RECEIPT_V1")
            self.assertEqual(receipt["sequence"], 1)
            self.assertEqual(receipt["operation"], "APPEND")
            self.assertEqual(receipt["pair"], "EUR_USD")
            self.assertEqual(receipt["cycle_id"], "cycle-1")

            bad_forecast = copy.copy(forecast)
            bad_context = copy.deepcopy(context)
            bad_context["execution"]["spread_band"] = "WIDE"
            bad_forecast.technical_context_v1 = bad_context
            with self.assertRaisesRegex(ValueError, "TECHNICAL_CONTEXT_HASH_MISMATCH"):
                record_forecast(bad_forecast, data_root=root, cycle_id="cycle-2")


if __name__ == "__main__":
    unittest.main()

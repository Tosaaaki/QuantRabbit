from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.strategy.regime_family_weighting import (
    build_regime_family_weighting_receipt,
    verify_regime_family_weighting_receipt,
)


def _chart(
    *,
    primary_regime: str = "TREND_STRONG",
    trend_score: float = 1.0,
    mean_reversion_score: float = -0.4,
    breakout_score: float = 0.8,
    session: str = "ASIA",
) -> dict:
    views = []
    for index, timeframe in enumerate(("D", "H4", "H1", "M30", "M15", "M5", "M1")):
        state = primary_regime if timeframe == "M15" else "TREND_WEAK"
        views.append(
            {
                "granularity": timeframe,
                "regime_reading": {
                    "state": state,
                    "confidence": 0.8,
                    "atr_percentile": 80.0 if timeframe == "M5" else 50.0,
                },
                "family_scores": {
                    "trend_score": trend_score,
                    "mean_rev_score": mean_reversion_score,
                    "breakout_score": breakout_score,
                    "breakout_components": {"donchian_break": 1.0},
                },
                "structure": {
                    "structure_events": [
                        {
                            "kind": "BOS_UP",
                            "index": index,
                            "close_confirmed": True,
                        }
                    ]
                },
            }
        )
    return {
        "pair": "EUR_USD",
        "generated_at_utc": "2026-07-13T00:00:00+00:00",
        "session": {"current_tag": session, "bucket": "NY"},
        "chart_story": "H1(TREND_STRONG ADX=30) H4(TREND_STRONG ADX=30)",
        "confluence": {"dominant_regime": "TREND_STRONG"},
        "views": views,
    }


def _sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _long_failed_break_candles() -> list[dict]:
    base = datetime(2026, 7, 13, tzinfo=timezone.utc)
    candles = [
        {
            "t": (base + timedelta(minutes=5 * index)).isoformat(),
            "o": 1.1000,
            "h": 1.1010,
            "l": 1.0990,
            "c": 1.1000,
            "complete": True,
        }
        for index in range(20)
    ]
    candles.append(
        {
            "t": "2026-07-13T01:40:00+00:00",
            "o": 1.0995,
            "h": 1.1005,
            "l": 1.0985,
            "c": 1.0992,
            "complete": True,
        }
    )
    return candles


class RegimeFamilyWeightingTest(unittest.TestCase):
    def test_primary_regime_switches_selected_technical_family_and_method(self) -> None:
        trend = build_regime_family_weighting_receipt(
            _chart(primary_regime="TREND_STRONG", trend_score=0.9),
            pair="EUR_USD",
        )
        range_receipt = build_regime_family_weighting_receipt(
            _chart(
                primary_regime="RANGE",
                trend_score=0.9,
                mean_reversion_score=0.8,
            ),
            pair="EUR_USD",
        )

        self.assertEqual(
            trend["by_timeframe"]["M15"]["selected_family"],
            "TREND",
        )
        self.assertEqual(
            trend["source_identity"]["selected_method"],
            "TREND_CONTINUATION",
        )
        self.assertEqual(
            range_receipt["by_timeframe"]["M15"]["selected_family"],
            "MEAN_REVERSION",
        )
        self.assertEqual(
            range_receipt["source_identity"]["selected_method"],
            "RANGE_ROTATION",
        )
        self.assertNotEqual(trend["weights"], range_receipt["weights"])
        self.assertEqual(
            verify_regime_family_weighting_receipt(trend, pair="EUR_USD"),
            (True, None),
        )
        self.assertEqual(
            verify_regime_family_weighting_receipt(
                range_receipt,
                pair="EUR_USD",
            ),
            (True, None),
        )

    def test_failed_break_selects_breakout_failure_only_from_exact_m5_proof(self) -> None:
        chart = _chart(primary_regime="BREAKOUT_PENDING")
        next(
            view for view in chart["views"] if view["granularity"] == "M5"
        )["recent_candles"] = _long_failed_break_candles()

        receipt = build_regime_family_weighting_receipt(chart, pair="EUR_USD")

        self.assertEqual(
            verify_regime_family_weighting_receipt(receipt, pair="EUR_USD"),
            (True, None),
        )
        self.assertEqual(
            receipt["source_identity"]["selected_method"],
            "BREAKOUT_FAILURE",
        )
        self.assertEqual(
            receipt["source_identity"]["failed_break_direction"],
            "LONG",
        )
        self.assertEqual(
            receipt["m5_failed_break_evidence"]["direction"],
            "LONG",
        )

        tampered = deepcopy(receipt)
        tampered["m5_failed_break_evidence"]["direction"] = "SHORT"
        tampered["m5_failed_break_evidence"]["evidence_sha256"] = _sha(
            {
                key: item
                for key, item in tampered["m5_failed_break_evidence"].items()
                if key != "evidence_sha256"
            }
        )
        tampered["source_identity"]["failed_break_direction"] = "SHORT"
        tampered["source_identity"]["failed_break_evidence_sha256"] = tampered[
            "m5_failed_break_evidence"
        ]["evidence_sha256"]
        tampered["source_identity_sha256"] = _sha(tampered["source_identity"])
        tampered["receipt_sha256"] = _sha(
            {key: item for key, item in tampered.items() if key != "receipt_sha256"}
        )
        self.assertEqual(
            verify_regime_family_weighting_receipt(tampered, pair="EUR_USD"),
            (False, "M5_FAILED_BREAK_EVIDENCE_DERIVATION_MISMATCH"),
        )

    def test_builder_uses_serialized_score_precision_for_direction_and_weight(self) -> None:
        receipt = build_regime_family_weighting_receipt(
            _chart(trend_score=0.123456789),
            pair="EUR_USD",
        )

        self.assertEqual(
            verify_regime_family_weighting_receipt(receipt, pair="EUR_USD"),
            (True, None),
        )
        self.assertEqual(receipt["by_timeframe"]["M15"]["selected_score"], 0.123457)

        rounded_zero = build_regime_family_weighting_receipt(
            _chart(trend_score=0.0000004),
            pair="EUR_USD",
        )
        self.assertEqual(
            verify_regime_family_weighting_receipt(rounded_zero, pair="EUR_USD"),
            (True, None),
        )
        self.assertEqual(rounded_zero["aggregate"]["direction"], "EITHER")
        self.assertEqual(
            rounded_zero["by_timeframe"]["M15"]["selected_direction"],
            "EITHER",
        )

    def test_current_tag_and_regime_method_change_weights_and_receipt_sha(self) -> None:
        asia = build_regime_family_weighting_receipt(_chart(), pair="EUR_USD")
        london_chart = _chart(session="LONDON")
        london = build_regime_family_weighting_receipt(london_chart, pair="EUR_USD")

        self.assertEqual(asia["source_identity"]["session"], "ASIA")
        self.assertEqual(asia["source_identity"]["situation"], "ASIA_TREND")
        self.assertEqual(
            asia["source_identity"]["family_selected_method"],
            "TREND_CONTINUATION",
        )
        self.assertNotEqual(asia["weights"], london["weights"])
        self.assertNotEqual(asia["receipt_sha256"], london["receipt_sha256"])
        self.assertEqual(
            verify_regime_family_weighting_receipt(asia, pair="EUR_USD"),
            (True, None),
        )

    def test_rehashed_arbitrary_weights_fail_pure_policy_recomputation(self) -> None:
        receipt = build_regime_family_weighting_receipt(_chart(), pair="EUR_USD")
        forged = deepcopy(receipt)
        forged["weights"]["D"], forged["weights"]["H4"] = (
            forged["weights"]["H4"],
            forged["weights"]["D"],
        )
        for timeframe, row in forged["by_timeframe"].items():
            old_weight = float(row["tf_weight"])
            directional_score = (
                float(row["weighted_directional_score"]) / old_weight
                if old_weight > 0.0
                else 0.0
            )
            new_weight = float(forged["weights"][timeframe])
            row["tf_weight"] = new_weight
            row["weighted_directional_score"] = round(
                new_weight * directional_score,
                12,
            )
        rows = forged["by_timeframe"].values()
        total = sum(float(row["weighted_directional_score"]) for row in rows)
        forged["aggregate"] = {
            "direction": "UP" if total > 0.0 else "DOWN" if total < 0.0 else "EITHER",
            "weighted_directional_score": round(total, 12),
            "directional_coverage_weight": min(
                1.0,
                round(
                    sum(
                        float(row["tf_weight"])
                        for row in forged["by_timeframe"].values()
                        if row["selected_direction"] in {"UP", "DOWN"}
                    ),
                    12,
                ),
            ),
            "selected_family_coverage_weight": min(
                1.0,
                round(
                    sum(
                        float(row["tf_weight"])
                        for row in forged["by_timeframe"].values()
                        if row["selected_family"] != "NONE"
                        and row["selected_score"] is not None
                    ),
                    12,
                ),
            ),
        }
        forged["receipt_sha256"] = _sha(
            {key: value for key, value in forged.items() if key != "receipt_sha256"}
        )

        self.assertEqual(
            verify_regime_family_weighting_receipt(forged, pair="EUR_USD"),
            (False, "REGIME_FAMILY_WEIGHTING_POLICY_WEIGHT_MISMATCH"),
        )

    def test_breakout_pending_is_non_directional_even_with_up_bos_and_donchian(self) -> None:
        receipt = build_regime_family_weighting_receipt(
            _chart(primary_regime="BREAKOUT_PENDING"),
            pair="EUR_USD",
        )

        self.assertEqual(receipt["source_identity"]["family_selected_method"], None)
        self.assertEqual(receipt["source_identity"]["selected_method"], None)
        m15 = receipt["by_timeframe"]["M15"]
        self.assertEqual(m15["selected_family"], "BREAKOUT")
        self.assertEqual(m15["selected_direction"], "EITHER")
        self.assertEqual(m15["weighted_directional_score"], 0.0)
        self.assertEqual(
            verify_regime_family_weighting_receipt(receipt, pair="EUR_USD"),
            (True, None),
        )

    def test_transition_primary_keeps_method_none_even_if_other_rows_point_up(self) -> None:
        receipt = build_regime_family_weighting_receipt(
            _chart(primary_regime="TRANSITION", trend_score=1.0),
            pair="EUR_USD",
        )

        self.assertIsNone(receipt["source_identity"]["selected_method"])
        self.assertEqual(receipt["aggregate"]["direction"], "UP")
        self.assertGreater(receipt["aggregate"]["directional_coverage_weight"], 0.0)
        self.assertEqual(
            verify_regime_family_weighting_receipt(receipt, pair="EUR_USD"),
            (True, None),
        )

    def test_news_override_records_effective_method_without_hiding_family_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calendar = Path(tmp) / "calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "impact": "HIGH",
                                "currency": "USD",
                                "timestamp_utc": "2026-07-13T00:10:00+00:00",
                                "title": "CPI",
                            }
                        ]
                    }
                )
            )
            receipt = build_regime_family_weighting_receipt(
                _chart(),
                pair="EUR_USD",
                calendar_path=calendar,
                now_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            )

        source = receipt["source_identity"]
        self.assertEqual(source["family_selected_method"], "TREND_CONTINUATION")
        self.assertEqual(source["effective_weight_method"], "EVENT_RISK")
        self.assertEqual(source["selected_method"], "EVENT_RISK")
        self.assertTrue(source["news_event_active"])
        self.assertEqual(
            receipt["policy_inputs"]["effective_weight_method"],
            "EVENT_RISK",
        )
        self.assertEqual(
            verify_regime_family_weighting_receipt(receipt, pair="EUR_USD"),
            (True, None),
        )


if __name__ == "__main__":
    unittest.main()

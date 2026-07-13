from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.strategy.price_action import aggregate_price_action_score
from quant_rabbit.strategy.tf_weights import BASELINE_WEIGHTS
from quant_rabbit.strategy.tf_weights import EDGE_MULT_HIGH
from quant_rabbit.strategy.tf_weights import build_dynamic_tf_policy_evidence
from quant_rabbit.strategy.tf_weights import derive_dynamic_tf_policy_from_evidence
from quant_rabbit.strategy.tf_weights import dynamic_tf_weights
from quant_rabbit.strategy.tf_weights import dynamic_tf_policy_evidence_sha256
from quant_rabbit.strategy.tf_weights import verify_dynamic_tf_policy_evidence


class DynamicTfWeightsCalendarTest(unittest.TestCase):
    @staticmethod
    def _rehash(evidence: dict[str, object]) -> None:
        evidence["evidence_sha256"] = dynamic_tf_policy_evidence_sha256(
            evidence
        )

    def test_raw_evidence_recomputes_classifier_and_all_seven_atr(self) -> None:
        now = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)
        chart = {
            "views": [
                {
                    "granularity": timeframe,
                    "regime_reading": {"atr_percentile": index * 10.0},
                }
                for index, timeframe in enumerate(BASELINE_WEIGHTS, start=1)
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            missing_calendar = Path(tmp) / "calendar-missing.json"
            missing_profile = Path(tmp) / "profile-missing.json"
            evidence = build_dynamic_tf_policy_evidence(
                session="ASIA",
                chart_story=(
                    "H1(TREND_STRONG ADX=31.0) "
                    "H4(TREND_WEAK ADX=26.0) "
                    "M15(RANGE ADX=12.0) M5(RANGE ADX=10.0)"
                ),
                dominant_regime="TREND_STRONG",
                method="TREND_CONTINUATION",
                pair="USD_JPY",
                pair_chart=chart,
                calendar_path=missing_calendar,
                strategy_profile_path=missing_profile,
                now_utc=now,
            )

            dynamic_result = dynamic_tf_weights(
                session="ASIA",
                chart_story=(
                    "H1(TREND_STRONG ADX=31.0) "
                    "H4(TREND_WEAK ADX=26.0) "
                    "M15(RANGE ADX=12.0) M5(RANGE ADX=10.0)"
                ),
                dominant_regime="TREND_STRONG",
                method="TREND_CONTINUATION",
                pair="USD_JPY",
                pair_chart=chart,
                calendar_path=missing_calendar,
                strategy_profile_path=missing_profile,
                now_utc=now,
                include_trace=True,
                include_evidence=True,
            )

        valid, error = verify_dynamic_tf_policy_evidence(evidence)
        self.assertTrue(valid, error)
        self.assertEqual(evidence["derived_situation"], "ASIA_TREND")
        self.assertEqual(
            evidence["classifier_inputs"]["timeframes"]["H1"],
            {"present": True, "regime": "TREND_STRONG", "adx": 31.0},
        )
        self.assertEqual(
            set(evidence["atr_percentile_by_timeframe"]),
            set(BASELINE_WEIGHTS),
        )
        policy, weights, label = derive_dynamic_tf_policy_from_evidence(evidence)
        self.assertEqual(policy["situation"], "ASIA_TREND")
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertIn("ASIA_TREND", label)
        self.assertEqual(dynamic_result[0], weights)
        self.assertEqual(dynamic_result[2], policy)
        self.assertEqual(dynamic_result[3], evidence)

    def test_raw_evidence_uses_same_indicator_atr_fallback_as_forecast_context(self) -> None:
        chart = {
            "views": [
                {
                    "granularity": timeframe,
                    "regime_reading": {"state": "TREND_WEAK"},
                    "indicators": {"atr_percentile_100": index / 10.0},
                }
                for index, timeframe in enumerate(BASELINE_WEIGHTS, start=1)
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            evidence = build_dynamic_tf_policy_evidence(
                session="ASIA",
                dominant_regime="TREND_WEAK",
                pair="EUR_USD",
                pair_chart=chart,
                calendar_path=Path(tmp) / "missing-calendar.json",
                strategy_profile_path=Path(tmp) / "missing-profile.json",
                now_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            )

        self.assertEqual(
            evidence["atr_percentile_by_timeframe"],
            {
                timeframe: round(index * 10.0, 4)
                for index, timeframe in enumerate(BASELINE_WEIGHTS, start=1)
            },
        )
        self.assertEqual(
            verify_dynamic_tf_policy_evidence(evidence),
            (True, None),
        )

    def test_classifier_tamper_rehashed_still_fails_derived_situation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = build_dynamic_tf_policy_evidence(
                session="ASIA",
                chart_story=(
                    "H1(TREND_STRONG ADX=31.0) H4(TREND_STRONG ADX=30.0)"
                ),
                dominant_regime="TREND_STRONG",
                pair="EUR_USD",
                calendar_path=Path(tmp) / "missing-calendar.json",
                strategy_profile_path=Path(tmp) / "missing-profile.json",
                now_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            )
        tampered = deepcopy(evidence)
        tampered["classifier_inputs"]["timeframes"]["H1"]["adx"] = 1.0
        tampered["classifier_inputs"]["timeframes"]["H4"]["adx"] = 1.0
        self._rehash(tampered)

        valid, error = verify_dynamic_tf_policy_evidence(tampered)

        self.assertFalse(valid)
        self.assertEqual(error, "DYNAMIC_TF_EVIDENCE_SITUATION_MISMATCH")

    def test_news_source_and_candidates_recompute_event_flag(self) -> None:
        now = datetime(2026, 6, 5, 12, 25, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            calendar = Path(tmp) / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                            }
                        ]
                    }
                )
            )
            expected_calendar_sha256 = hashlib.sha256(
                calendar.read_bytes()
            ).hexdigest()
            evidence = build_dynamic_tf_policy_evidence(
                session="ASIA",
                dominant_regime="RANGE",
                method="RANGE_ROTATION",
                pair="GBP_USD",
                calendar_path=calendar,
                strategy_profile_path=Path(tmp) / "missing-profile.json",
                now_utc=now,
            )

        news = evidence["news_evidence"]
        self.assertEqual(
            news["source"]["sha256"],
            expected_calendar_sha256,
        )
        self.assertEqual(len(news["candidates"]), 1)
        self.assertTrue(news["active"])
        policy, _, label = derive_dynamic_tf_policy_from_evidence(evidence)
        self.assertEqual(policy["effective_weight_method"], "EVENT_RISK")
        self.assertEqual(policy["situation"], "NY_OVERLAP")
        self.assertIn("news:USD:Non-Farm Employment Change 5m", label)

        tampered = deepcopy(evidence)
        tampered["news_evidence"]["active"] = False
        tampered["news_evidence"]["reason"] = ""
        self._rehash(tampered)
        valid, error = verify_dynamic_tf_policy_evidence(tampered)
        self.assertFalse(valid)
        self.assertEqual(error, "DYNAMIC_TF_EVIDENCE_NEWS_MISMATCH")

    def test_selected_strategy_source_recomputes_allowlisted_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "strategy_profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "method": "TREND_CONTINUATION",
                                "positive_evidence_n": 101,
                                "live_net_jpy": 700.0,
                            },
                            {
                                "pair": "EUR_USD",
                                "method": "TREND_CONTINUATION",
                                "positive_evidence_n": 110,
                                "live_net_jpy": 800.0,
                            },
                        ]
                    }
                )
            )
            evidence = build_dynamic_tf_policy_evidence(
                dominant_regime="TREND_STRONG",
                method="TREND_CONTINUATION",
                pair="EUR_USD",
                calendar_path=Path(tmp) / "missing-calendar.json",
                strategy_profile_path=profile,
                now_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            )

        strategy = evidence["strategy_profile_evidence"]
        self.assertEqual(len(strategy["candidates"]), 2)
        self.assertEqual(strategy["selected"]["source_index"], 1)
        self.assertEqual(strategy["multiplier"], EDGE_MULT_HIGH)
        self.assertIn("n=110/live=800", strategy["label"])

        tampered = deepcopy(evidence)
        tampered["strategy_profile_evidence"]["multiplier"] = 1.0
        self._rehash(tampered)
        valid, error = verify_dynamic_tf_policy_evidence(tampered)
        self.assertFalse(valid)
        self.assertEqual(error, "DYNAMIC_TF_EVIDENCE_STRATEGY_MISMATCH")

        forged_selection = deepcopy(evidence)
        forged_selection["strategy_profile_evidence"]["candidates"][0][
            "positive_evidence_n"
        ] = 1_000
        self._rehash(forged_selection)
        valid, error = verify_dynamic_tf_policy_evidence(forged_selection)
        self.assertFalse(valid)
        self.assertEqual(
            error,
            "DYNAMIC_TF_EVIDENCE_STRATEGY_SELECTION_MISMATCH",
        )

    def test_candidate_overflow_is_bounded_and_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = root / "calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-07-13T00:00:00+00:00",
                                "currency": "USD",
                                "impact": "HIGH",
                                "title": f"event-{index}",
                            }
                            for index in range(20)
                        ]
                    }
                )
            )
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "method": "TREND_CONTINUATION",
                                "positive_evidence_n": 500 + index,
                                "live_net_jpy": 1_000.0,
                            }
                            for index in range(20)
                        ]
                    }
                )
            )
            evidence = build_dynamic_tf_policy_evidence(
                session="ASIA",
                dominant_regime="TREND_STRONG",
                method="TREND_CONTINUATION",
                pair="EUR_USD",
                calendar_path=calendar,
                strategy_profile_path=profile,
                now_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            )

        self.assertEqual(
            evidence["news_evidence"]["source"]["status"],
            "LIMIT_EXCEEDED",
        )
        self.assertEqual(evidence["news_evidence"]["candidates"], [])
        self.assertTrue(evidence["news_evidence"]["active"])
        strategy = evidence["strategy_profile_evidence"]
        self.assertEqual(strategy["source"]["status"], "LIMIT_EXCEEDED")
        self.assertEqual(strategy["candidates"], [])
        self.assertIsNone(strategy["selected"])
        self.assertEqual(strategy["multiplier"], 1.0)
        self.assertEqual(verify_dynamic_tf_policy_evidence(evidence), (True, None))
        self.assertLessEqual(
            len(
                json.dumps(
                    evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            4600,
        )

    def test_utf8_packet_overflow_forces_event_risk_and_neutral_strategy(self) -> None:
        now = datetime(2026, 7, 13, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_root = root
            for index in range(4):
                long_root /= f"{index}-" + ("x" * 198)
            long_root.mkdir(parents=True)
            calendar = long_root / "calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": now.isoformat(),
                                "currency": "USD",
                                "impact": "HIGH",
                                "title": "\U0001f4a5" * 120,
                            }
                            for _ in range(4)
                        ]
                    }
                )
            )
            profile = long_root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "method": "TREND_CONTINUATION",
                                "positive_evidence_n": 500 + index,
                                "live_net_jpy": 1_000.0,
                            }
                            for index in range(4)
                        ]
                    }
                )
            )

            evidence = build_dynamic_tf_policy_evidence(
                session="ASIA",
                dominant_regime="TREND_STRONG",
                method="TREND_CONTINUATION",
                pair="EUR_USD",
                calendar_path=calendar,
                strategy_profile_path=profile,
                now_utc=now,
            )

        news = evidence["news_evidence"]
        self.assertEqual(news["source"]["status"], "LIMIT_EXCEEDED")
        self.assertEqual(news["candidates"], [])
        self.assertTrue(news["active"])
        self.assertEqual(news["reason"], "calendar_unavailable")
        strategy = evidence["strategy_profile_evidence"]
        self.assertEqual(strategy["source"]["status"], "LIMIT_EXCEEDED")
        self.assertEqual(strategy["candidates"], [])
        self.assertIsNone(strategy["selected"])
        self.assertEqual(strategy["multiplier"], 1.0)
        self.assertEqual(strategy["label"], "")
        policy, _, _ = derive_dynamic_tf_policy_from_evidence(evidence)
        self.assertTrue(policy["news_event_active"])
        self.assertEqual(policy["effective_weight_method"], "EVENT_RISK")
        self.assertEqual(verify_dynamic_tf_policy_evidence(evidence), (True, None))
        self.assertLessEqual(
            len(
                json.dumps(
                    evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            4600,
        )

    def test_packet_overflow_after_fail_closed_compaction_still_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_missing_root = root
            for _ in range(10):
                long_missing_root /= "\U0001f4a5" * 50

            with self.assertRaisesRegex(
                ValueError,
                "DYNAMIC_TF_EVIDENCE_LIMIT_EXCEEDED",
            ):
                build_dynamic_tf_policy_evidence(
                    method="TREND_CONTINUATION",
                    pair="EUR_USD",
                    calendar_path=long_missing_root / "calendar.json",
                    strategy_profile_path=long_missing_root / "profile.json",
                    now_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
                )

    def test_missing_one_atr_timeframe_rehashed_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = build_dynamic_tf_policy_evidence(
                pair="EUR_USD",
                calendar_path=Path(tmp) / "missing-calendar.json",
                strategy_profile_path=Path(tmp) / "missing-profile.json",
                now_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            )
        tampered = deepcopy(evidence)
        del tampered["atr_percentile_by_timeframe"]["M1"]
        self._rehash(tampered)

        valid, error = verify_dynamic_tf_policy_evidence(tampered)

        self.assertFalse(valid)
        self.assertEqual(error, "DYNAMIC_TF_EVIDENCE_ATR_INVALID")

    def test_regime_atr_percentile_keeps_zero_to_one_values_on_percent_scale(self) -> None:
        _, _, trace = dynamic_tf_weights(
            session="ASIA",
            dominant_regime="RANGE",
            method="RANGE_ROTATION",
            pair="EUR_USD",
            pair_chart={
                "views": [
                    {
                        "granularity": "M5",
                        "regime_reading": {"atr_percentile": 0.5},
                    }
                ]
            },
            include_trace=True,
        )

        self.assertEqual(
            trace["atr_percentile_by_timeframe"]["M5"],
            0.5,
        )

    def test_timestamp_utc_calendar_event_triggers_event_risk_overlay(self) -> None:
        now = datetime(2026, 6, 5, 12, 25, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "economic_calendar.json"
            path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                            }
                        ]
                    }
                )
            )

            _, label = dynamic_tf_weights(
                session="ASIA_RANGE",
                dominant_regime="RANGE",
                method="RANGE_ROTATION",
                pair="GBP_USD",
                calendar_path=path,
                now_utc=now,
            )

        self.assertIn("NY_OVERLAP", label)
        self.assertIn("news:USD:Non-Farm Employment Change", label)

    def test_missing_calendar_feed_triggers_event_risk_overlay(self) -> None:
        now = datetime(2026, 6, 5, 13, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "economic_calendar.json"
            path.write_text(
                json.dumps(
                    {
                        "events": [],
                        "issues": ["MISSING_FOREX_FACTORY_FEED: HTTP Error 429: Too Many Requests"],
                    }
                )
            )

            _, label = dynamic_tf_weights(
                session="ASIA_RANGE",
                dominant_regime="RANGE",
                method="RANGE_ROTATION",
                pair="GBP_USD",
                calendar_path=path,
                now_utc=now,
            )

        self.assertIn("NY_OVERLAP", label)
        self.assertIn("news:calendar_unavailable", label)

    def test_price_action_uses_caller_verified_receipt_weights(self) -> None:
        chart = {
            "pair": "EUR_USD",
            "views": [
                {"granularity": "H4"},
                {"granularity": "M5"},
            ],
        }
        with patch(
            "quant_rabbit.strategy.price_action.read_timeframe",
            side_effect=lambda view, *_args: view["granularity"],
        ), patch(
            "quant_rabbit.strategy.price_action.price_action_score",
            side_effect=lambda read, _side: (
                (10.0, ["H4 proof"])
                if read == "H4"
                else (-10.0, ["M5 counter"])
            ),
        ):
            score, reasons = aggregate_price_action_score(
                chart,
                "LONG",
                1.1,
                10_000.0,
                method="TREND_CONTINUATION",
                receipt_weights={
                    timeframe: 1.0 if timeframe == "H4" else 0.0
                    for timeframe in BASELINE_WEIGHTS
                },
                receipt_label="ASIA_TREND_RECEIPT",
            )

        self.assertEqual(score, 10.0)
        self.assertEqual(reasons[0], "TF weighting=ASIA_TREND_RECEIPT")


if __name__ == "__main__":
    unittest.main()

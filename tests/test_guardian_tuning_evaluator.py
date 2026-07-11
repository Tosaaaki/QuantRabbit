from __future__ import annotations

import json
import unittest

from quant_rabbit.guardian_tuning_evaluator import (
    FIXED_ACCEPTANCE_THRESHOLD,
    METRIC_NAMES,
    OBJECTIVE,
    PRIMARY_METRIC,
    derive_result,
    evaluate_threshold_cohort,
    source_identity,
    source_semantic_digest,
    validate_source,
)
from tools.guardian_tuning_metric_evaluator import evaluate as evaluate_frozen


def _cohort(*, count: int = 20) -> dict:
    review_completed = "2026-07-09T00:00:00+00:00"
    lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
    return {
        "schema_version": 5,
        "cohort_id": "forward-eurusd-forecast-confidence-2026w27",
        "source_watermark": {
            "selection_cutoff_utc": "2026-07-11T23:59:00+00:00",
            "last_oanda_transaction_id": "473024",
            "ledger_rowid_watermark": 500,
            "ledger_prefix_sha256": "a" * 64,
            "canonical_outcome_set_sha256": "b" * 64,
            "entry_thesis_prefix_bytes": 100,
            "entry_thesis_prefix_sha256": "c" * 64,
            "forecast_history_prefix_bytes": 200,
            "forecast_history_prefix_sha256": "d" * 64,
        },
        "selection_cutoff_utc": "2026-07-11T23:59:00+00:00",
        "pair": "EUR_USD",
        "bot_family": "forecast",
        "lane_id": lane,
        "parameter": "forecast_confidence_floor",
        "validation_contract": {
            "mode": "FORWARD_POST_REVIEW",
            "review_digest_sha256": "e" * 64,
            "review_completed_at_utc": review_completed,
            "minimum_sample_count": 20,
        },
        "provenance": {
            "generator": "guardian_tuning_cohort_builder_v3",
            "execution_ledger_coverage_start_utc": "2026-05-06T16:52:01+00:00",
            "last_oanda_transaction_id": "473024",
            "post_cost_financing_included": True,
        },
        "samples": [
            {
                "sample_id": f"sample-{index}",
                "pair": "EUR_USD",
                "bot_family": "forecast",
                "lane_id": lane,
                "trade_id": f"trade-{index}",
                "order_id": f"order-{index}",
                "entry_at_utc": f"2026-07-10T{index:02d}:00:00+00:00",
                "closed_at_utc": f"2026-07-10T{index:02d}:30:00+00:00",
                "signal_observed_at_utc": "2026-07-09T23:59:59+00:00",
                "signal_record_sha256": f"{index:064x}",
                "signal_value": 0.50 if index < 4 else 0.60,
                "realized_net_jpy": -100.0 if index < 4 else 100.0,
                "entry_units": 1000.0,
                "net_jpy_per_1000_units": -100.0 if index < 4 else 100.0,
            }
            for index in range(count)
        ],
    }


class GuardianTuningEvaluatorTest(unittest.TestCase):
    def test_oanda_nanosecond_timestamps_normalize_to_microseconds(self) -> None:
        payload = _cohort()
        cutoff = "2026-07-11T23:59:00.123456789Z"
        payload["selection_cutoff_utc"] = cutoff
        payload["source_watermark"]["selection_cutoff_utc"] = cutoff
        payload["validation_contract"]["review_completed_at_utc"] = (
            "2026-07-09T00:00:00.000000001Z"
        )
        payload["provenance"]["execution_ledger_coverage_start_utc"] = (
            "2026-05-06T16:52:01.987654321Z"
        )
        payload["samples"][0]["signal_observed_at_utc"] = (
            "2026-07-09T23:59:59.123456789Z"
        )
        payload["samples"][0]["entry_at_utc"] = (
            "2026-07-10T00:00:00.234567891Z"
        )
        payload["samples"][0]["closed_at_utc"] = (
            "2026-07-10T00:30:00.345678912Z"
        )

        identity = validate_source(payload)

        self.assertEqual(
            identity["selection_cutoff_utc"],
            "2026-07-11T23:59:00.123456+00:00",
        )

    def test_frozen_self_contained_evaluator_matches_prepare_time_contract_logic(self) -> None:
        payload = _cohort()
        shared = evaluate_threshold_cohort(
            payload,
            parameter="forecast_confidence_floor",
            current_value=0.50,
            candidate_value=0.55,
        )
        frozen = evaluate_frozen(
            payload,
            parameter="forecast_confidence_floor",
            current_value=0.50,
            candidate_value=0.55,
            primary_metric=PRIMARY_METRIC,
            objective=OBJECTIVE,
            acceptance_threshold=FIXED_ACCEPTANCE_THRESHOLD,
        )
        result, improvement = derive_result(
            shared,
            primary_metric=PRIMARY_METRIC,
            objective=OBJECTIVE,
            acceptance_threshold=FIXED_ACCEPTANCE_THRESHOLD,
        )

        self.assertEqual(shared["baseline_metrics"], frozen["baseline_metrics"])
        self.assertEqual(shared["candidate_metrics"], frozen["candidate_metrics"])
        self.assertEqual(shared["acceptance_constraints"], frozen["acceptance_constraints"])
        self.assertEqual((result, improvement), (frozen["derived_result"], frozen["improvement"]))

    def test_raw_cohort_derives_metrics_after_threshold_values_are_bound(self) -> None:
        payload = _cohort()

        evaluation = evaluate_threshold_cohort(
            payload,
            parameter="forecast_confidence_floor",
            current_value=0.50,
            candidate_value=0.90,
        )

        self.assertEqual(evaluation["metric_names"], list(METRIC_NAMES))
        self.assertEqual(evaluation["baseline_metrics"]["trade_count"], 20)
        self.assertEqual(evaluation["candidate_metrics"]["trade_count"], 0)
        self.assertFalse(
            evaluation["acceptance_constraints"]["candidate_trade_count_sufficient"]
        )

    def test_frequency_collapse_cannot_be_accepted_even_when_payoff_improves(self) -> None:
        evaluation = evaluate_threshold_cohort(
            _cohort(),
            parameter="forecast_confidence_floor",
            current_value=0.50,
            candidate_value=0.90,
        )

        result, improvement = derive_result(
            evaluation,
            primary_metric=PRIMARY_METRIC,
            objective=OBJECTIVE,
            acceptance_threshold=FIXED_ACCEPTANCE_THRESHOLD,
        )

        self.assertLess(improvement, 0.0)
        self.assertEqual(result, "REJECTED_NO_IMPROVEMENT")

    def test_improvement_equal_to_threshold_is_rejected(self) -> None:
        payload = _cohort()
        for sample in payload["samples"]:
            sample["signal_value"] = 0.60
        evaluation = evaluate_threshold_cohort(
            payload,
            parameter="forecast_confidence_floor",
            current_value=0.55,
            candidate_value=0.59,
        )
        baseline = evaluation["baseline_metrics"][PRIMARY_METRIC]
        candidate = evaluation["candidate_metrics"][PRIMARY_METRIC]
        self.assertEqual(candidate - baseline, 0.0)

        result, _ = derive_result(
            evaluation,
            primary_metric=PRIMARY_METRIC,
            objective=OBJECTIVE,
            acceptance_threshold=FIXED_ACCEPTANCE_THRESHOLD,
        )

        self.assertEqual(result, "REJECTED_NO_IMPROVEMENT")

    def test_actual_baseline_includes_below_base_floor_support_exceptions(self) -> None:
        payload = _cohort()
        payload["samples"][0]["signal_value"] = 0.50
        payload["samples"][0]["realized_net_jpy"] = 1_000.0
        payload["samples"][0]["net_jpy_per_1000_units"] = 1_000.0
        for index in range(1, 4):
            payload["samples"][index]["signal_value"] = 0.68
            payload["samples"][index]["realized_net_jpy"] = -100.0
            payload["samples"][index]["net_jpy_per_1000_units"] = -100.0
        for index in range(4, 20):
            payload["samples"][index]["signal_value"] = 0.75
            payload["samples"][index]["realized_net_jpy"] = 10.0
            payload["samples"][index]["net_jpy_per_1000_units"] = 10.0

        evaluation = evaluate_threshold_cohort(
            payload,
            parameter="forecast_confidence_floor",
            current_value=0.65,
            candidate_value=0.70,
        )
        result, improvement = derive_result(
            evaluation,
            primary_metric=PRIMARY_METRIC,
            objective=OBJECTIVE,
            acceptance_threshold=FIXED_ACCEPTANCE_THRESHOLD,
        )

        self.assertEqual(evaluation["baseline_metrics"]["trade_count"], 20)
        self.assertEqual(evaluation["candidate_metrics"]["trade_count"], 16)
        self.assertLess(improvement, 0.0)
        self.assertEqual(result, "REJECTED_NO_IMPROVEMENT")

    def test_source_rejects_precomputed_or_extra_result_fields(self) -> None:
        payload = _cohort()
        payload["samples"][0]["candidate_metrics"] = {"expectancy_jpy": 9999.0}

        with self.assertRaisesRegex(ValueError, "field"):
            source_identity(payload)

    def test_signal_at_entry_timestamp_is_not_pre_entry_evidence(self) -> None:
        payload = _cohort()
        payload["samples"][0]["signal_observed_at_utc"] = payload["samples"][0][
            "entry_at_utc"
        ]

        with self.assertRaisesRegex(ValueError, "sample close must be valid"):
            source_identity(payload)

    def test_parameter_must_match_frozen_cohort(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_threshold_cohort(
                _cohort(),
                parameter="forecast_score_floor",
                current_value=0.50,
                candidate_value=0.55,
            )

    def test_no_repeat_digest_ignores_labels_refs_and_json_formatting(self) -> None:
        first = _cohort()
        second = json.loads(json.dumps(first, indent=7))
        second["cohort_id"] = "different-human-label"
        second["source_watermark"]["entry_thesis_prefix_sha256"] = "f" * 64
        second["source_watermark"]["forecast_history_prefix_sha256"] = "0" * 64

        self.assertEqual(
            source_semantic_digest(first),
            source_semantic_digest(second),
        )


if __name__ == "__main__":
    unittest.main()

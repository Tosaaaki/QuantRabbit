from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools import guardian_tuning_evidence_builder as builder
from tools import guardian_wake_dispatcher as dispatcher


class GuardianTuningEvidenceBuilderTest(unittest.TestCase):
    def test_forward_precommit_run_seal_terminal_and_fresh_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "data" / "guardian_tuning_work_order.json"
            now = datetime.now(timezone.utc)
            event = {
                "event_id": "builder-e2e",
                "event_type": "TECHNICAL_STATE_CHANGE",
                "pair": "EUR_USD",
                "direction": "SHORT",
                "thesis": "closed candle changed",
                "price_zone": "mid=1.170000 spread_pips=1.0",
                "severity": "P2",
                "recommended_review_type": "TUNING_REVIEW",
                "dedupe_key": "EUR_USD|technical|TECHNICAL_STATE_CHANGE|NO_ACTION",
                "action_hint": "NO_ACTION",
                "thesis_state": "ALIVE",
                "detected_at_utc": now.isoformat(),
                "wake_reason_codes": ["REGIME_STATE_CHANGE"],
                "details": {},
            }
            review = {
                "review_status": "TEST_REQUIRED",
                "affected_pairs": ["EUR_USD"],
                "affected_bot_families": ["forecast"],
                "hypothesis": "a higher recorded forecast floor improves forward capture",
                "falsifiable_experiment": "evaluate entries opened only after this review",
                "proposed_adjustments": [
                    {
                        "pair": "EUR_USD",
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                        "bot_family": "forecast",
                        "parameter": "forecast_confidence_floor",
                        "current_value": 0.65,
                        "candidate_value": 0.70,
                        "rationale": "one forward confidence-floor change",
                    }
                ],
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            }
            created = dispatcher._maybe_write_tuning_work_order(
                path=queue,
                selected_event=event,
                receipt={"bot_tuning_review": review},
                now=now,
            )
            lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
            cutoff = now + timedelta(microseconds=40)
            cohort = {
                "schema_version": 5,
                "cohort_id": "builder-e2e-forward-cohort",
                "source_watermark": {
                    "selection_cutoff_utc": cutoff.isoformat(),
                    "last_oanda_transaction_id": "473024",
                    "ledger_rowid_watermark": 500,
                    "ledger_prefix_sha256": "a" * 64,
                    "canonical_outcome_set_sha256": "b" * 64,
                    "entry_thesis_prefix_bytes": 100,
                    "entry_thesis_prefix_sha256": "c" * 64,
                    "forecast_history_prefix_bytes": 200,
                    "forecast_history_prefix_sha256": "d" * 64,
                },
                "selection_cutoff_utc": cutoff.isoformat(),
                "pair": "EUR_USD",
                "bot_family": "forecast",
                "lane_id": lane,
                "parameter": "forecast_confidence_floor",
                "validation_contract": {
                    "mode": "FORWARD_POST_REVIEW",
                    "review_digest_sha256": dispatcher._tuning_review_digest(review),
                    "review_completed_at_utc": now.isoformat(),
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
                        "entry_at_utc": (now + timedelta(microseconds=index + 1)).isoformat(),
                        "closed_at_utc": (now + timedelta(microseconds=index + 21)).isoformat(),
                        "signal_observed_at_utc": (now + timedelta(microseconds=index)).isoformat(),
                        "signal_record_sha256": f"{index:064x}",
                        "signal_value": 0.65 if index < 4 else 0.75,
                        "realized_net_jpy": -10.0 if index < 4 else 10.0,
                        "entry_units": 1000.0,
                        "net_jpy_per_1000_units": -10.0 if index < 4 else 10.0,
                    }
                    for index in range(20)
                ],
            }
            cohort_raw = (
                json.dumps(cohort, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode()
            cohort_digest = hashlib.sha256(cohort_raw).hexdigest()
            cohort_path = root / "data" / "guardian_tuning_cohorts" / f"{cohort_digest}.json"
            cohort_path.parent.mkdir(parents=True)
            cohort_path.write_bytes(cohort_raw)
            approved = root / "tools" / "guardian_tuning_metric_evaluator.py"
            approved.parent.mkdir(parents=True)
            approved.write_bytes(builder.APPROVED_EVALUATOR.read_bytes())

            canonical = {"status": "VALID"}
            with (
                patch.object(builder, "ROOT", root),
                patch.object(builder, "APPROVED_EVALUATOR", approved),
                patch.object(builder, "validate_canonical_forward_cohort", return_value=canonical),
                patch.object(dispatcher, "validate_canonical_forward_cohort", return_value=canonical),
                patch.object(
                    dispatcher,
                    "current_canonical_forward_source_tip",
                    return_value={
                        "last_oanda_transaction_id": "473024",
                        "ledger_rowid_watermark": 500,
                        "ledger_prefix_sha256": "a" * 64,
                        "entry_thesis_prefix_bytes": 100,
                        "entry_thesis_prefix_sha256": "c" * 64,
                        "forecast_history_prefix_bytes": 200,
                        "forecast_history_prefix_sha256": "d" * 64,
                    },
                ),
                patch.object(
                    dispatcher,
                    "current_execution_ledger_anchor",
                    return_value={
                        "ledger_rowid_watermark": 500,
                        "ledger_prefix_sha256": "a" * 64,
                        "execution_ledger_coverage_start_utc": (
                            "2026-05-06T16:52:01+00:00"
                        ),
                        "last_oanda_transaction_id": "473024",
                        "captured_at_utc": now.isoformat(),
                    },
                ),
            ):
                initialized = builder._init_run(
                    argparse.Namespace(
                        path=queue,
                        work_order_id=created["work_order_id"],
                        expected_observation_id=created["observation_id"],
                        experiment_id="builder-e2e-experiment",
                        source_data=cohort_path,
                        evaluator_artifact=approved,
                        cohort_id=cohort["cohort_id"],
                        source_watermark_json=json.dumps(cohort["source_watermark"]),
                        sample_count=len(cohort["samples"]),
                        evaluator=builder.EVALUATOR_NAME,
                        primary_metric=builder.PRIMARY_METRIC,
                        objective=builder.OBJECTIVE,
                        acceptance_threshold=builder.FIXED_ACCEPTANCE_THRESHOLD,
                        prepared_by="builder-e2e-test",
                        output=None,
                    )
                )
                self.assertEqual(initialized["status"], "EXPERIMENT_RUN_TEMPLATE_WRITTEN")
                sealed = builder._run(
                    argparse.Namespace(
                        path=queue,
                        work_order_id=created["work_order_id"],
                        expected_observation_id=created["observation_id"],
                        experiment_id="builder-e2e-experiment",
                        output=None,
                    )
                )
                self.assertEqual(sealed["status"], "EXPERIMENT_EVIDENCE_SEALED")
                terminal = dispatcher.transition_tuning_work_order(
                    path=queue,
                    work_order_id=created["work_order_id"],
                    expected_observation_id=created["observation_id"],
                    status="CONSUMED",
                    consumed_by="builder-e2e-test",
                    experiment_id="builder-e2e-experiment",
                    experiment_result=sealed["experiment_result"],
                    experiment_evidence_ref=sealed["experiment_evidence_ref"],
                    now=datetime.now(timezone.utc),
                )
                self.assertEqual(terminal["status"], "WORK_ORDER_TERMINAL_WRITTEN")
                self.assertEqual(
                    terminal["tuning_override_application"]["status"],
                    "OVERRIDE_STAGED",
                )
                self.assertEqual(
                    terminal["tuning_override_confirmation"]["status"],
                    "OVERRIDE_ACTIVATED",
                )
                self.assertTrue(
                    (root / "data" / "guardian_tuning_overrides.json").is_file()
                )
                reloaded = dispatcher._load_tuning_work_order(queue)
                self.assertNotIn("_read_error", reloaded)
                self.assertEqual(reloaded["terminal_history_count"], 1)


if __name__ == "__main__":
    unittest.main()

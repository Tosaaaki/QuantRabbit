from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from quant_rabbit.guardian_tuning_overrides import write_terminal_commitment_manifest
from tools import guardian_tuning_evidence_builder as builder
from tools import guardian_tuning_metric_evaluator as frozen_evaluator
from tools import guardian_wake_dispatcher as dispatcher


class GuardianTuningEvidenceSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.queue_path = self.root / "data" / "guardian_tuning_work_order.json"
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.work_order_generated_at = datetime(2026, 7, 11, 0, 0, tzinfo=timezone.utc)
        self.prepared_at = self.work_order_generated_at + timedelta(minutes=10)
        self.executed_at = self.prepared_at + timedelta(minutes=10)
        self.now = self.executed_at + timedelta(minutes=1)
        self.work_order_id = "guardian-tuning-security-test"
        self.observation_id = "observation-security-test"
        self.experiment_id = "experiment-security-test"
        self.semantic_state_id = "semantic-security-test"
        self.review = {
            "review_status": "TEST_REQUIRED",
            "affected_pairs": ["EUR_USD"],
            "affected_bot_families": ["forecast"],
            "hypothesis": "the candidate threshold improves the frozen cohort metric",
            "falsifiable_experiment": "compare the frozen baseline and candidate cohort",
            "proposed_adjustments": [
                {
                    "pair": "EUR_USD",
                    "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                    "bot_family": "forecast",
                    "parameter": "forecast_confidence_floor",
                    "current_value": 0.65,
                    "candidate_value": 0.70,
                    "rationale": "evaluate one bounded parameter change",
                }
            ],
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        }
        canonical = patch.object(
            dispatcher,
            "validate_canonical_forward_cohort",
            return_value={"status": "VALID"},
        )
        canonical.start()
        self.addCleanup(canonical.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_artifact(
        self,
        relative: Path,
        raw: bytes,
        *,
        content_addressed: bool,
        suffix: str = ".json",
    ) -> tuple[Path, str]:
        digest = hashlib.sha256(raw).hexdigest()
        if content_addressed:
            relative = relative / f"{digest}{suffix}"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path, f"{relative}#sha256={digest}"

    @staticmethod
    def _json_bytes(payload: dict[str, Any]) -> bytes:
        return (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    def _frozen_inputs(
        self,
        *,
        excluded_realized_net_jpy: int | float = -10.0,
        equal_selection: bool = False,
    ) -> tuple[dict[str, Any], str, str, dict[str, Any], str]:
        source_payload = {
            "schema_version": 5,
            "cohort_id": "frozen-security-cohort",
            "source_watermark": {
                "selection_cutoff_utc": "2026-07-11T00:21:00+00:00",
                "last_oanda_transaction_id": "473024",
                "ledger_rowid_watermark": 500,
                "ledger_prefix_sha256": "a" * 64,
                "canonical_outcome_set_sha256": "b" * 64,
                "entry_thesis_prefix_bytes": 100,
                "entry_thesis_prefix_sha256": "c" * 64,
                "forecast_history_prefix_bytes": 200,
                "forecast_history_prefix_sha256": "d" * 64,
            },
            "selection_cutoff_utc": "2026-07-11T00:21:00+00:00",
            "pair": "EUR_USD",
            "bot_family": "forecast",
            "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
            "parameter": "forecast_confidence_floor",
            "validation_contract": {
                "mode": "FORWARD_POST_REVIEW",
                "review_digest_sha256": dispatcher._tuning_review_digest(self.review),
                "review_completed_at_utc": self.work_order_generated_at.isoformat(),
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
                    "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                    "trade_id": f"trade-{index}",
                    "order_id": f"order-{index}",
                    "entry_at_utc": (
                        self.work_order_generated_at + timedelta(minutes=index + 1)
                    ).isoformat(),
                    "closed_at_utc": (
                        self.work_order_generated_at + timedelta(minutes=index + 1, seconds=30)
                    ).isoformat(),
                    "signal_observed_at_utc": (
                        self.work_order_generated_at
                        + timedelta(minutes=index + 1, seconds=-1)
                    ).isoformat(),
                    "signal_record_sha256": f"{index:064x}",
                    "signal_value": 0.75 if equal_selection or index >= 4 else 0.65,
                    "realized_net_jpy": (
                        excluded_realized_net_jpy
                        if index == 0
                        else -10.0 if index < 4 else 20.0
                    ),
                    "entry_units": 1000.0,
                    "net_jpy_per_1000_units": (
                        excluded_realized_net_jpy
                        if index == 0
                        else -10.0 if index < 4 else 20.0
                    ),
                }
                for index in range(20)
            ],
        }
        source_path, source_ref = self._write_artifact(
            Path("data/guardian_tuning_experiment_inputs/data"),
            self._json_bytes(source_payload),
            content_addressed=True,
        )
        evaluator_raw = builder.APPROVED_EVALUATOR.read_bytes()
        _, evaluator_ref = self._write_artifact(
            Path("data/guardian_tuning_experiment_inputs/evaluators"),
            evaluator_raw,
            content_addressed=True,
            suffix=".py",
        )
        evaluation = frozen_evaluator.evaluate(
            source_payload,
            parameter="forecast_confidence_floor",
            current_value=0.65,
            candidate_value=0.70,
            primary_metric=builder.PRIMARY_METRIC,
            objective=builder.OBJECTIVE,
            acceptance_threshold=builder.FIXED_ACCEPTANCE_THRESHOLD,
        )
        evaluator_stdout = (
            json.dumps(evaluation, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        return source_payload, source_ref, evaluator_ref, evaluation, hashlib.sha256(
            evaluator_stdout
        ).hexdigest()

    def _prepared_contract(
        self,
        *,
        source_payload: dict[str, Any],
        source_ref: str,
        evaluator_ref: str,
        threshold: float = builder.FIXED_ACCEPTANCE_THRESHOLD,
    ) -> dict[str, Any]:
        material = {
            "semantic_state_id": self.semantic_state_id,
            "review_digest_sha256": dispatcher._tuning_review_digest(self.review),
            "cohort_id": source_payload["cohort_id"],
            "source_watermark": source_payload["source_watermark"],
            "sample_count": len(source_payload["samples"]),
            "evaluator": builder.EVALUATOR_NAME,
            "source_data_ref": source_ref,
            "evaluator_artifact_ref": evaluator_ref,
            "primary_metric": builder.PRIMARY_METRIC,
            "objective": builder.OBJECTIVE,
            "acceptance_threshold": threshold,
            "metric_names": sorted(builder.METRIC_NAMES),
            "active_parameter_binding": {
                "parameter": "forecast_confidence_floor",
                "environment_variable": "QR_FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE",
                "resolved_value": 0.65,
                "method": "TREND_CONTINUATION",
            },
            "source_identity": {
                "cohort_id": source_payload["cohort_id"],
                "source_watermark": source_payload["source_watermark"],
                "selection_cutoff_utc": source_payload["selection_cutoff_utc"],
                "pair": source_payload["pair"],
                "bot_family": source_payload["bot_family"],
                "lane_id": source_payload["lane_id"],
                "parameter": source_payload["parameter"],
                "sample_count": len(source_payload["samples"]),
                "validation_contract": source_payload["validation_contract"],
            },
        }
        digest = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        source_semantic_digest = dispatcher.tuning_source_semantic_digest(source_payload)
        evaluator_sha256 = evaluator_ref.rsplit("=", 1)[-1]
        semantic_digest = dispatcher._tuning_experiment_semantic_digest(
            adjustment=self.review["proposed_adjustments"][0],
            source_semantic_digest=source_semantic_digest,
            evaluator_artifact_sha256=evaluator_sha256,
            acceptance_threshold=threshold,
        )
        self.assertIsNotNone(semantic_digest)
        return {
            **material,
            "status": "PREPARED",
            "experiment_id": self.experiment_id,
            "observation_id": self.observation_id,
            "experiment_contract_digest": digest,
            "experiment_semantic_digest": semantic_digest,
            "source_semantic_digest": source_semantic_digest,
            "evaluator_artifact_sha256": evaluator_sha256,
            "prepared_at_utc": self.prepared_at.isoformat(),
            "prepared_by": "security-test",
        }

    def _run_payload(
        self,
        *,
        contract: dict[str, Any],
        evaluation: dict[str, Any],
        stdout_sha256: str,
        result: str = "ACCEPTED_IMPROVEMENT",
        executed_at: datetime | None = None,
    ) -> dict[str, Any]:
        timestamp = executed_at or self.executed_at
        adjustment = self.review["proposed_adjustments"][0]
        source_sha256 = str(contract["source_data_ref"]).rsplit("=", 1)[-1]
        evaluator_sha256 = str(contract["evaluator_artifact_ref"]).rsplit("=", 1)[-1]
        return {
            "schema_version": 1,
            "status": "COMPLETED",
            "exit_status": (
                "COMPLETED_SUCCESS"
                if result == "ACCEPTED_IMPROVEMENT"
                else "COMPLETED_NO_EDGE"
            ),
            "work_order_id": self.work_order_id,
            "observation_id": self.observation_id,
            "experiment_id": self.experiment_id,
            "experiment_contract_digest": contract["experiment_contract_digest"],
            "review_digest_sha256": dispatcher._tuning_review_digest(self.review),
            "pair": adjustment["pair"],
            "bot_family": adjustment["bot_family"],
            "parameter": adjustment["parameter"],
            "current_value": adjustment["current_value"],
            "candidate_value": adjustment["candidate_value"],
            "cohort_id": evaluation["cohort_id"],
            "source_watermark": evaluation["source_watermark"],
            "sample_count": evaluation["sample_count"],
            "baseline_metrics": evaluation["baseline_metrics"],
            "candidate_metrics": evaluation["candidate_metrics"],
            "acceptance_constraints": evaluation["acceptance_constraints"],
            "evaluator": contract["evaluator"],
            "source_data_ref": contract["source_data_ref"],
            "evaluator_artifact_ref": contract["evaluator_artifact_ref"],
            "primary_metric": contract["primary_metric"],
            "objective": contract["objective"],
            "acceptance_threshold": contract["acceptance_threshold"],
            "result": result,
            "generated_at_utc": timestamp.isoformat(),
            "evaluator_execution": {
                "runner": builder.RUNNER_NAME,
                "exit_code": 0,
                "stdout_sha256": stdout_sha256,
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "source_data_sha256": source_sha256,
                "evaluator_artifact_sha256": evaluator_sha256,
                "executed_at_utc": timestamp.isoformat(),
            },
            "no_live_side_effects": True,
        }

    def _write_run(
        self,
        payload: dict[str, Any],
        *,
        content_addressed: bool = True,
    ) -> str:
        relative = Path("data/guardian_tuning_experiment_runs")
        if not content_addressed:
            relative /= "mutable-run.json"
        _, ref = self._write_artifact(
            relative,
            self._json_bytes(payload),
            content_addressed=content_addressed,
        )
        return ref

    def _validate_run(
        self,
        run_ref: str,
        contract: dict[str, Any],
        *,
        result: str = "ACCEPTED_IMPROVEMENT",
    ) -> dict[str, Any]:
        return dispatcher._validate_tuning_experiment_run_ref(
            queue_path=self.queue_path,
            source_artifact_ref=run_ref,
            work_order_id=self.work_order_id,
            observation_id=self.observation_id,
            experiment_id=self.experiment_id,
            experiment_result=result,
            review=self.review,
            semantic_state_id=self.semantic_state_id,
            prepared_contract=contract,
            work_order_generated_at=self.work_order_generated_at.isoformat(),
            review_completed_at_utc=self.work_order_generated_at.isoformat(),
            now=self.now,
        )

    def _evidence_payload(
        self,
        *,
        run_ref: str,
        contract: dict[str, Any],
        result: str = "ACCEPTED_IMPROVEMENT",
    ) -> dict[str, Any]:
        adjustment = self.review["proposed_adjustments"][0]
        return {
            "schema_version": 1,
            "status": "COMPLETED",
            "work_order_id": self.work_order_id,
            "observation_id": self.observation_id,
            "experiment_id": self.experiment_id,
            "review_digest_sha256": dispatcher._tuning_review_digest(self.review),
            "hypothesis": self.review["hypothesis"],
            "falsifiable_experiment": self.review["falsifiable_experiment"],
            "pair": adjustment["pair"],
            "bot_family": adjustment["bot_family"],
            "parameter": adjustment["parameter"],
            "current_value": adjustment["current_value"],
            "candidate_value": adjustment["candidate_value"],
            "result": result,
            "source_artifact_ref": run_ref,
            "experiment_contract_digest": contract["experiment_contract_digest"],
            "generated_at_utc": self.executed_at.isoformat(),
            "no_live_side_effects": True,
        }

    def _validate_evidence(
        self,
        evidence_ref: str,
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        return dispatcher._validate_tuning_experiment_evidence_ref(
            queue_path=self.queue_path,
            evidence_ref=evidence_ref,
            work_order_id=self.work_order_id,
            observation_id=self.observation_id,
            experiment_id=self.experiment_id,
            experiment_result="ACCEPTED_IMPROVEMENT",
            review=self.review,
            semantic_state_id=self.semantic_state_id,
            prepared_contract=contract,
            work_order_generated_at=self.work_order_generated_at.isoformat(),
            review_completed_at_utc=self.work_order_generated_at.isoformat(),
            now=self.now,
        )

    def test_run_requires_exact_evaluator_execution_provenance(self) -> None:
        source, source_ref, evaluator_ref, evaluation, stdout_sha = self._frozen_inputs()
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=evaluator_ref,
        )
        valid_payload = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
        )
        valid_ref = self._write_run(valid_payload)
        self.assertEqual(self._validate_run(valid_ref, contract)["status"], "VALID")

        for field, value in (
            ("evaluator_execution", None),
            ("runner", "forged-runner"),
            ("source_data_sha256", "0" * 64),
            ("evaluator_artifact_sha256", "0" * 64),
            ("stdout_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                forged = json.loads(json.dumps(valid_payload))
                if field == "evaluator_execution":
                    forged.pop(field)
                else:
                    forged["evaluator_execution"][field] = value
                forged_ref = self._write_run(forged)
                self.assertNotEqual(
                    self._validate_run(forged_ref, contract)["status"],
                    "VALID",
                )

    def test_prepared_source_is_bound_to_queue_review_time_and_reviewed_lane(self) -> None:
        source, source_ref, _, _, _ = self._frozen_inputs()

        def validate(payload: dict[str, Any], source_artifact_ref: str) -> dict[str, Any]:
            source_validation = dispatcher._validate_project_artifact_ref(
                queue_path=self.queue_path,
                artifact_ref=source_artifact_ref,
                allowed_roots=("data/guardian_tuning_experiment_inputs/data",),
                max_bytes=dispatcher.MAX_TUNING_SOURCE_BYTES,
                require_content_addressed=True,
            )
            return dispatcher._validate_prepared_tuning_source(
                queue_path=self.queue_path,
                source_validation=source_validation,
                adjustment=self.review["proposed_adjustments"][0],
                review=self.review,
                review_completed_at_utc=self.work_order_generated_at.isoformat(),
                cohort_id=payload["cohort_id"],
                source_watermark=payload["source_watermark"],
                sample_count=len(payload["samples"]),
                require_current_tips=False,
            )

        self.assertEqual(validate(source, source_ref)["status"], "VALID")

        backdated = json.loads(json.dumps(source))
        backdated["validation_contract"]["review_completed_at_utc"] = (
            self.work_order_generated_at - timedelta(days=1)
        ).isoformat()
        _, backdated_ref = self._write_artifact(
            Path("data/guardian_tuning_experiment_inputs/data"),
            self._json_bytes(backdated),
            content_addressed=True,
        )
        backdated_result = validate(backdated, backdated_ref)
        self.assertEqual(backdated_result["status"], "SOURCE_CONTRACT_MISMATCH")
        self.assertIn(
            "review_completed_at_utc",
            backdated_result["mismatched_fields"],
        )

        reselected = json.loads(json.dumps(source))
        reselected["lane_id"] = (
            "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:LIMIT"
        )
        for sample in reselected["samples"]:
            sample["lane_id"] = reselected["lane_id"]
        _, reselected_ref = self._write_artifact(
            Path("data/guardian_tuning_experiment_inputs/data"),
            self._json_bytes(reselected),
            content_addressed=True,
        )
        reselected_result = validate(reselected, reselected_ref)
        self.assertEqual(reselected_result["status"], "SOURCE_CONTRACT_MISMATCH")
        self.assertIn("lane_id", reselected_result["mismatched_fields"])

    def test_initial_prepare_requires_current_ledger_and_log_tips(self) -> None:
        source, source_ref, _, _, _ = self._frozen_inputs()
        source_validation = dispatcher._validate_project_artifact_ref(
            queue_path=self.queue_path,
            artifact_ref=source_ref,
            allowed_roots=("data/guardian_tuning_experiment_inputs/data",),
            max_bytes=dispatcher.MAX_TUNING_SOURCE_BYTES,
            require_content_addressed=True,
        )
        source_tip = {
            key: source["source_watermark"][key]
            for key in (
                "last_oanda_transaction_id",
                "ledger_rowid_watermark",
                "ledger_prefix_sha256",
                "entry_thesis_prefix_bytes",
                "entry_thesis_prefix_sha256",
                "forecast_history_prefix_bytes",
                "forecast_history_prefix_sha256",
            )
        }

        with patch.object(
            dispatcher,
            "current_canonical_forward_source_tip",
            return_value=source_tip,
        ):
            valid = dispatcher._validate_prepared_tuning_source(
                queue_path=self.queue_path,
                source_validation=source_validation,
                adjustment=self.review["proposed_adjustments"][0],
                review=self.review,
                review_completed_at_utc=self.work_order_generated_at.isoformat(),
                cohort_id=source["cohort_id"],
                source_watermark=source["source_watermark"],
                sample_count=len(source["samples"]),
                require_current_tips=True,
            )
        stale_tip = {**source_tip, "ledger_rowid_watermark": 501}
        with patch.object(
            dispatcher,
            "current_canonical_forward_source_tip",
            return_value=stale_tip,
        ):
            stale = dispatcher._validate_prepared_tuning_source(
                queue_path=self.queue_path,
                source_validation=source_validation,
                adjustment=self.review["proposed_adjustments"][0],
                review=self.review,
                review_completed_at_utc=self.work_order_generated_at.isoformat(),
                cohort_id=source["cohort_id"],
                source_watermark=source["source_watermark"],
                sample_count=len(source["samples"]),
                require_current_tips=True,
            )

        self.assertEqual(valid["status"], "VALID")
        self.assertEqual(stale["status"], "SOURCE_NOT_CURRENT_TIP")
        self.assertIn("ledger_rowid_watermark", stale["mismatched_fields"])

    def test_run_must_be_executed_after_contract_preparation(self) -> None:
        source, source_ref, evaluator_ref, evaluation, stdout_sha = self._frozen_inputs()
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=evaluator_ref,
        )
        predating = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
            executed_at=self.prepared_at - timedelta(seconds=1),
        )
        predating_ref = self._write_run(predating)
        self.assertNotEqual(
            self._validate_run(predating_ref, contract)["status"],
            "VALID",
        )

    def test_improvement_equal_to_threshold_is_rejected(self) -> None:
        # Excluding one -5 JPY sample improves net JPY per opportunity by
        # exactly 1.0, which is equal to (not greater than) the threshold.
        source, source_ref, evaluator_ref, evaluation, stdout_sha = self._frozen_inputs(
            equal_selection=True,
        )
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=evaluator_ref,
            threshold=builder.FIXED_ACCEPTANCE_THRESHOLD,
        )
        falsely_accepted = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
            result="ACCEPTED_IMPROVEMENT",
        )
        run_ref = self._write_run(falsely_accepted)
        self.assertNotEqual(
            self._validate_run(run_ref, contract)["status"],
            "VALID",
        )

    def test_non_finite_conversion_overflow_fails_closed(self) -> None:
        source, source_ref, evaluator_ref, evaluation, stdout_sha = self._frozen_inputs()
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=evaluator_ref,
        )
        overflow = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
        )
        overflow["candidate_metrics"][builder.PRIMARY_METRIC] = 10**400
        run_ref = self._write_run(overflow)
        validation = self._validate_run(run_ref, contract)
        self.assertNotEqual(validation["status"], "VALID")

    def test_reported_metrics_are_recomputed_from_frozen_cohort(self) -> None:
        source, source_ref, evaluator_ref, evaluation, stdout_sha = self._frozen_inputs()
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=evaluator_ref,
        )
        legitimate = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
        )
        legitimate_ref = self._write_run(legitimate)
        self.assertEqual(self._validate_run(legitimate_ref, contract)["status"], "VALID")

        forged = json.loads(json.dumps(legitimate))
        forged["candidate_metrics"][builder.PRIMARY_METRIC] = 999999.0
        forged_ref = self._write_run(forged)
        self.assertNotEqual(
            self._validate_run(forged_ref, contract)["status"],
            "VALID",
        )

    def test_run_ref_must_be_content_addressed(self) -> None:
        source, source_ref, evaluator_ref, evaluation, stdout_sha = self._frozen_inputs()
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=evaluator_ref,
        )
        run_payload = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
        )
        mutable_run_ref = self._write_run(run_payload, content_addressed=False)
        self.assertNotEqual(
            self._validate_run(mutable_run_ref, contract)["status"],
            "VALID",
        )

    def test_evidence_ref_must_be_content_addressed(self) -> None:
        source, source_ref, evaluator_ref, evaluation, stdout_sha = self._frozen_inputs()
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=evaluator_ref,
        )
        run_payload = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
        )
        immutable_run_ref = self._write_run(run_payload)
        evidence_payload = self._evidence_payload(
            run_ref=immutable_run_ref,
            contract=contract,
        )
        immutable_evidence_path, immutable_evidence_ref = self._write_artifact(
            Path("data/guardian_tuning_evidence"),
            self._json_bytes(evidence_payload),
            content_addressed=True,
        )
        self.assertTrue(immutable_evidence_path.is_file())
        self.assertEqual(
            self._validate_evidence(immutable_evidence_ref, contract)["status"],
            "VALID",
        )

        _, mutable_evidence_ref = self._write_artifact(
            Path("data/guardian_tuning_evidence/mutable-evidence.json"),
            self._json_bytes(evidence_payload),
            content_addressed=False,
        )
        self.assertNotEqual(
            self._validate_evidence(mutable_evidence_ref, contract)["status"],
            "VALID",
        )

    def test_fresh_load_rejects_malicious_evaluator_without_executing_it(self) -> None:
        source, source_ref, _, evaluation, stdout_sha = self._frozen_inputs()
        execution_marker = self.root / "malicious-evaluator-executed.txt"
        malicious_evaluator = (
            "from pathlib import Path\n"
            f"Path({str(execution_marker)!r}).write_text('executed')\n"
            "def evaluate(*args, **kwargs):\n"
            "    raise RuntimeError('data-path Python must never execute')\n"
        ).encode("utf-8")
        _, malicious_evaluator_ref = self._write_artifact(
            Path("data/guardian_tuning_experiment_inputs/evaluators"),
            malicious_evaluator,
            content_addressed=True,
            suffix=".py",
        )
        contract = self._prepared_contract(
            source_payload=source,
            source_ref=source_ref,
            evaluator_ref=malicious_evaluator_ref,
        )
        run_payload = self._run_payload(
            contract=contract,
            evaluation=evaluation,
            stdout_sha256=stdout_sha,
        )
        run_ref = self._write_run(run_payload)
        direct_validation = self._validate_run(run_ref, contract)
        self.assertEqual(
            direct_validation["status"],
            "SOURCE_EVALUATOR_NOT_CURRENTLY_APPROVED",
        )
        self.assertFalse(execution_marker.exists())
        evidence_payload = self._evidence_payload(
            run_ref=run_ref,
            contract=contract,
        )
        _, evidence_ref = self._write_artifact(
            Path("data/guardian_tuning_evidence"),
            self._json_bytes(evidence_payload),
            content_addressed=True,
        )
        terminal = {
            "generated_at_utc": self.work_order_generated_at.isoformat(),
            "work_order_id": self.work_order_id,
            "event_fingerprint": self.observation_id,
            "observation_id": self.observation_id,
            "latest_observation_id": self.observation_id,
            "semantic_state_id": self.semantic_state_id,
            "status": "CONSUMED",
            "consumed_at_utc": self.executed_at.isoformat(),
            "consumed_by": "security-test",
            "experiment_id": self.experiment_id,
            "experiment_result": "ACCEPTED_IMPROVEMENT",
            "experiment_evidence_ref": evidence_ref,
            "experiment_contract_digest": contract["experiment_contract_digest"],
            "experiment_semantic_digest": contract["experiment_semantic_digest"],
            "prepared_experiment_contract": contract,
            "bot_tuning_review": self.review,
            "terminal_transition_source": "guardian_tuning_work_order_lifecycle",
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        }
        terminal_digest = hashlib.sha256(
            json.dumps(
                terminal,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        terminal_record_ref = write_terminal_commitment_manifest(
            queue_path=self.queue_path,
            terminal=terminal,
        )
        queue_payload = {
            **terminal,
            "schema_version": 2,
            "queue_schema_revision": dispatcher.TUNING_QUEUE_SCHEMA_REVISION,
            "work_orders": [],
            "pending_count": 0,
            "terminal_history": [terminal],
            "terminal_history_count": 1,
            "experiment_semantic_digest_history": [
                contract["experiment_semantic_digest"]
            ],
            "experiment_id_digest_history": [
                dispatcher._experiment_id_digest(self.experiment_id)
            ],
            "override_lifecycle_heads": [
                {
                    "status": "ACTIVE_COMMITTED",
                    "override_key": (
                        "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
                        "|forecast_confidence_floor"
                    ),
                    "work_order_id": self.work_order_id,
                    "experiment_id": self.experiment_id,
                    "experiment_result": "ACCEPTED_IMPROVEMENT",
                    "experiment_evidence_ref": evidence_ref,
                    "experiment_contract_digest": contract[
                        "experiment_contract_digest"
                    ],
                    "terminal_confirmation_sha256": terminal_digest,
                    "terminal_record_ref": terminal_record_ref,
                    "pair": "EUR_USD",
                    "method": "TREND_CONTINUATION",
                    "lane_id": (
                        "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
                    ),
                    "parameter": "forecast_confidence_floor",
                    "candidate_value": 0.70,
                    "committed_at_utc": self.executed_at.isoformat(),
                    "live_permission_allowed": False,
                    "no_direct_oanda": True,
                }
            ],
        }
        self.queue_path.write_bytes(self._json_bytes(queue_payload))

        loaded = dispatcher._load_tuning_work_order(self.queue_path)

        self.assertIn("_read_error", loaded)
        self.assertFalse(execution_marker.exists())


if __name__ == "__main__":
    unittest.main()

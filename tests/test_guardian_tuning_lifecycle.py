from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools import guardian_wake_dispatcher as dispatcher


NOW = datetime(2026, 7, 11, 1, 0, tzinfo=timezone.utc)


def _review(*, candidate: float) -> dict:
    return {
        "review_status": "TEST_REQUIRED",
        "affected_pairs": ["EUR_USD"],
        "affected_bot_families": ["forecast"],
        "hypothesis": f"entry floor {candidate} changes normalized post-cost capture",
        "falsifiable_experiment": "evaluate the complete frozen canonical cohort",
        "proposed_adjustments": [
            {
                "pair": "EUR_USD",
                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                "bot_family": "forecast",
                "parameter": "forecast_confidence_floor",
                "current_value": 0.65,
                "candidate_value": candidate,
                "rationale": "one bounded score-floor test",
            }
        ],
        "live_permission_allowed": False,
        "no_direct_oanda": True,
        "preserve_blockers": True,
    }


def _event() -> dict:
    return {
        "event_id": "technical-lifecycle",
        "event_type": "TECHNICAL_STATE_CHANGE",
        "pair": "EUR_USD",
        "direction": "SHORT",
        "thesis": "closed candle state changed",
        "price_zone": "mid=1.170000 spread_pips=1.0",
        "severity": "P2",
        "recommended_review_type": "TUNING_REVIEW",
        "dedupe_key": "EUR_USD|technical|TECHNICAL_STATE_CHANGE|NO_ACTION",
        "action_hint": "NO_ACTION",
        "thesis_state": "ALIVE",
        "detected_at_utc": NOW.isoformat(),
        "wake_reason_codes": ["REGIME_STATE_CHANGE"],
        "details": {
            "mid": 1.17,
            "closed_candle_watermarks": {
                "M1": "2026-07-11T00:59:00+00:00",
                "M5": "2026-07-11T00:55:00+00:00",
                "M15": "2026-07-11T00:45:00+00:00",
            },
        },
    }


def _write_ref(root: Path, relative_root: Path, raw: bytes, suffix: str) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    relative = relative_root / f"{digest}{suffix}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return f"{relative}#sha256={digest}"


class GuardianTuningLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        def canonical(payload: dict, **kwargs: object) -> dict:
            review = kwargs.get("review")
            expected = dispatcher._tuning_review_digest(review) if isinstance(review, dict) else ""
            observed = str(
                (payload.get("validation_contract") or {}).get("review_digest_sha256")
            )
            return (
                {"status": "VALID"}
                if observed == expected
                else {"status": "CANONICAL_COHORT_MISMATCH"}
            )

        mocked = patch.object(
            dispatcher,
            "validate_canonical_forward_cohort",
            side_effect=canonical,
        )
        mocked.start()
        self.addCleanup(mocked.stop)
        current_tip = patch.object(
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
        )
        current_tip.start()
        self.addCleanup(current_tip.stop)

    def _inputs(
        self,
        root: Path,
        *,
        review: dict,
        reviewed_at: datetime,
    ) -> tuple[str, str, dict]:
        lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
        cutoff = reviewed_at + timedelta(microseconds=40)
        source = {
            "schema_version": 5,
            "cohort_id": "lifecycle-cohort",
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
                "review_completed_at_utc": reviewed_at.isoformat(),
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
                    "entry_at_utc": (reviewed_at + timedelta(microseconds=index + 1)).isoformat(),
                    "closed_at_utc": (reviewed_at + timedelta(microseconds=index + 21)).isoformat(),
                    "signal_observed_at_utc": (reviewed_at + timedelta(microseconds=index)).isoformat(),
                    "signal_record_sha256": f"{index:064x}",
                    "signal_value": 0.70 if index < 4 else 0.80,
                    "realized_net_jpy": 10.0 if index % 2 else -10.0,
                    "entry_units": 1000.0,
                    "net_jpy_per_1000_units": 10.0 if index % 2 else -10.0,
                }
                for index in range(20)
            ],
        }
        source_raw = (
            json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        evaluator_raw = (
            Path(dispatcher.__file__).resolve().with_name(
                "guardian_tuning_metric_evaluator.py"
            ).read_bytes()
        )
        return (
            _write_ref(
                root,
                Path("data/guardian_tuning_experiment_inputs/data"),
                source_raw,
                ".json",
            ),
            _write_ref(
                root,
                Path("data/guardian_tuning_experiment_inputs/evaluators"),
                evaluator_raw,
                ".py",
            ),
            source,
        )

    def _prepare(
        self,
        *,
        path: Path,
        work_order_id: str,
        observation_id: str,
        experiment_id: str,
        source_ref: str,
        evaluator_ref: str,
        source: dict,
        now: datetime,
    ) -> dict:
        return dispatcher.prepare_tuning_experiment_contract(
            path=path,
            work_order_id=work_order_id,
            expected_observation_id=observation_id,
            experiment_id=experiment_id,
            cohort_id=source["cohort_id"],
            source_watermark=source["source_watermark"],
            sample_count=len(source["samples"]),
            evaluator=dispatcher.TUNING_EVALUATOR_NAME,
            source_data_ref=source_ref,
            evaluator_artifact_ref=evaluator_ref,
            primary_metric=dispatcher.TUNING_EVALUATOR_PRIMARY_METRIC,
            objective=dispatcher.TUNING_EVALUATOR_OBJECTIVE,
            acceptance_threshold=dispatcher.TUNING_FIXED_ACCEPTANCE_THRESHOLD,
            metric_names=list(dispatcher.TUNING_EVALUATOR_METRIC_NAMES),
            prepared_by="lifecycle-test",
            now=now,
        )

    def test_abort_requires_reacquisition_and_new_forward_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "guardian_tuning_work_order.json"
            created = dispatcher._maybe_write_tuning_work_order(
                path=path,
                selected_event=_event(),
                receipt={"bot_tuning_review": _review(candidate=0.70)},
                now=NOW,
            )
            initial_review = _review(candidate=0.70)
            source_ref, evaluator_ref, source = self._inputs(
                root,
                review=initial_review,
                reviewed_at=NOW,
            )
            prepared = self._prepare(
                path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="experiment-1",
                source_ref=source_ref,
                evaluator_ref=evaluator_ref,
                source=source,
                now=NOW + timedelta(seconds=1),
            )
            self.assertEqual(prepared["status"], "EXPERIMENT_CONTRACT_PREPARED")
            contract = prepared["prepared_experiment_contract"]
            failure = {
                "schema_version": 1,
                "status": "ABORTED",
                "work_order_id": created["work_order_id"],
                "observation_id": created["observation_id"],
                "experiment_id": "experiment-1",
                "experiment_contract_digest": contract["experiment_contract_digest"],
                "experiment_semantic_digest": contract["experiment_semantic_digest"],
                "aborted_by": "lifecycle-test",
                "reason": "approved evaluator process failed",
                "generated_at_utc": (NOW + timedelta(seconds=2)).isoformat(),
                "no_live_side_effects": True,
            }
            failure_ref = _write_ref(
                root,
                Path("data/guardian_tuning_experiment_failures"),
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
                ".json",
            )
            aborted = dispatcher.abort_tuning_experiment_contract(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                experiment_id="experiment-1",
                aborted_by="lifecycle-test",
                reason="approved evaluator process failed",
                failure_evidence_ref=failure_ref,
                now=NOW + timedelta(seconds=2),
            )
            self.assertEqual(aborted["status"], "EXPERIMENT_CONTRACT_ABORTED")

            same_adjustment_review = _review(candidate=0.70)
            same_adjustment_review["hypothesis"] = (
                "different prose cannot relabel the same experiment"
            )
            rebound = dispatcher.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                review=same_adjustment_review,
                reviewed_by="lifecycle-test",
                now=NOW + timedelta(seconds=3),
            )
            self.assertEqual(rebound["status"], "WORK_ORDER_REVIEW_ENRICHED")
            stale_repeated = self._prepare(
                path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="experiment-2",
                source_ref=source_ref,
                evaluator_ref=evaluator_ref,
                source=source,
                now=NOW + timedelta(seconds=4),
            )
            self.assertEqual(stale_repeated["status"], "EXPERIMENT_CONTRACT_INVALID")

            changed = dispatcher.enrich_tuning_work_order_review(
                path=path,
                work_order_id=created["work_order_id"],
                expected_observation_id=created["observation_id"],
                review=_review(candidate=0.75),
                reviewed_by="lifecycle-test",
                now=NOW + timedelta(seconds=5),
            )
            self.assertEqual(changed["status"], "WORK_ORDER_REVIEW_ENRICHED")
            stale_changed_prepare = self._prepare(
                path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="experiment-3",
                source_ref=source_ref,
                evaluator_ref=evaluator_ref,
                source=source,
                now=NOW + timedelta(seconds=6),
            )
            self.assertEqual(
                stale_changed_prepare["status"],
                "EXPERIMENT_CONTRACT_INVALID",
            )
            refreshed_review = _review(candidate=0.75)
            new_source_ref, new_evaluator_ref, new_source = self._inputs(
                root,
                review=refreshed_review,
                reviewed_at=NOW + timedelta(seconds=5),
            )
            changed_prepare = self._prepare(
                path=path,
                work_order_id=created["work_order_id"],
                observation_id=created["observation_id"],
                experiment_id="experiment-4",
                source_ref=new_source_ref,
                evaluator_ref=new_evaluator_ref,
                source=new_source,
                now=NOW + timedelta(seconds=7),
            )
            self.assertEqual(changed_prepare["status"], "EXPERIMENT_CONTRACT_PREPARED")


if __name__ == "__main__":
    unittest.main()

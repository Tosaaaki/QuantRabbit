from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit import guardian_tuning_monitor as monitor
from quant_rabbit import guardian_tuning_overrides as overrides
from tools import guardian_wake_dispatcher as dispatcher
from tools import guardian_tuning_post_activation_monitor as monitor_tool
from tools.guardian_tuning_post_activation_monitor import run_monitor


LANE = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
PRIMARY_METRIC = "net_jpy_per_1000_units_per_opportunity"


def _review() -> dict:
    return {
        "review_status": "TEST_REQUIRED",
        "proposed_adjustments": [
            {
                "pair": "EUR_USD",
                "lane_id": LANE,
                "bot_family": "forecast",
                "parameter": "forecast_confidence_floor",
                "current_value": 0.65,
                "candidate_value": 0.70,
            }
        ],
    }


def _contract() -> dict:
    return {
        "experiment_contract_digest": "a" * 64,
        "source_identity": {"lane_id": LANE},
        "active_parameter_binding": {
            "parameter": "forecast_confidence_floor",
            "resolved_value": 0.65,
            "method": "TREND_CONTINUATION",
        },
    }


def _activation_ledger_anchor() -> dict:
    return {
        "ledger_rowid_watermark": 1,
        "ledger_prefix_sha256": "a" * 64,
        "execution_ledger_coverage_start_utc": "2026-01-01T00:00:00+00:00",
        "last_oanda_transaction_id": "1",
        "captured_at_utc": "2026-07-11T00:00:00+00:00",
    }


def _write_terminal_queue(
    *,
    queue_path: Path,
    application: dict,
    experiment_id: str,
    evidence_ref: str,
) -> None:
    terminal = {
        "work_order_id": "work-order-monitor",
        "status": "CONSUMED",
        "experiment_id": experiment_id,
        "experiment_result": "ACCEPTED_IMPROVEMENT",
        "experiment_evidence_ref": evidence_ref,
        "experiment_contract_digest": "a" * 64,
        "terminal_transition_source": "guardian_tuning_work_order_lifecycle",
        "tuning_override_application": application,
        "bot_tuning_review": _review(),
        "prepared_experiment_contract": _contract(),
        "live_permission_allowed": False,
        "no_direct_oanda": True,
    }
    digest = overrides._canonical_terminal_digest(terminal)
    terminal_ref = overrides.write_terminal_commitment_manifest(
        queue_path=queue_path,
        terminal=terminal,
    )
    head = {
        "status": "ACTIVE_COMMITTED",
        "override_key": application["override_key"],
        "work_order_id": "work-order-monitor",
        "experiment_id": experiment_id,
        "experiment_result": "ACCEPTED_IMPROVEMENT",
        "experiment_evidence_ref": evidence_ref,
        "experiment_contract_digest": "a" * 64,
        "terminal_confirmation_sha256": digest,
        "terminal_record_ref": terminal_ref,
        "pair": "EUR_USD",
        "method": "TREND_CONTINUATION",
        "lane_id": LANE,
        "parameter": "forecast_confidence_floor",
        "candidate_value": 0.70,
        "activated_at_utc": application["activated_at_utc"],
        "activation_ledger_anchor": application["activation_ledger_anchor"],
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_permission_allowed": False,
        "no_direct_oanda": True,
    }
    queue = {
        **terminal,
        "schema_version": 2,
        "queue_schema_revision": 4,
        "work_orders": [],
        "pending_count": 0,
        "terminal_history": [terminal],
        "terminal_history_count": 1,
        "experiment_semantic_digest_history": [],
        "experiment_id_digest_history": [],
        "override_lifecycle_heads": [head],
    }
    queue_path.write_text(json.dumps(queue))


def _write_hand_authored_monitor_evidence(root: Path, payload: dict) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    directory = root / "data" / monitor.MONITOR_EVIDENCE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{digest}.json").write_bytes(raw)
    return (
        f"data/{monitor.MONITOR_EVIDENCE_DIRECTORY}/{digest}.json"
        f"#sha256={digest}"
    )


class GuardianTuningMonitorTest(unittest.TestCase):
    def test_monitor_paths_must_share_one_canonical_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            ledger_path = root / "data" / "execution_ledger.db"
            with self.assertRaisesRegex(ValueError, "overrides must be the queue sibling"):
                run_monitor(
                    queue_path=queue_path,
                    override_path=root / "other" / "guardian_tuning_overrides.json",
                    ledger_path=ledger_path,
                    now=datetime.now(timezone.utc),
                )

    def test_existing_valid_evidence_is_reused_after_a_commit_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            ledger_path = root / "data" / "execution_ledger.db"
            queue_path.parent.mkdir(parents=True)
            digest = "7" * 64
            evidence_path = (
                root
                / "data"
                / "guardian_tuning_monitor_evidence"
                / f"{digest}.json"
            )
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text("{}\n")
            validation = {
                "status": "VALID",
                "decision": "KEEP",
                "primary_metric_value": 1.5,
            }
            with patch.object(
                monitor_tool,
                "validate_post_activation_monitor_evidence",
                return_value=validation,
            ) as validate:
                reused = monitor_tool._existing_valid_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=ledger_path,
                    record={"experiment_id": "experiment-monitor"},
                )

            self.assertEqual(reused["decision"], "KEEP")
            self.assertEqual(reused["primary_metric_value"], 1.5)
            self.assertEqual(validate.call_count, 1)

    def test_stale_matching_evidence_is_skipped_but_other_invalid_evidence_blocks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            ledger_path = root / "data" / "execution_ledger.db"
            evidence_path = (
                root
                / "data"
                / "guardian_tuning_monitor_evidence"
                / ("8" * 64 + ".json")
            )
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text("{}\n")
            stale = {
                "status": "MONITOR_EVIDENCE_COHORT_INVALID",
                "cohort_validation": {
                    "status": "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
                },
            }
            with patch.object(
                monitor_tool,
                "validate_post_activation_monitor_evidence",
                return_value=stale,
            ):
                reusable = monitor_tool._existing_valid_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=ledger_path,
                    record={"experiment_id": "experiment-monitor"},
                )
            self.assertIsNone(reusable)

            with (
                patch.object(
                    monitor_tool,
                    "validate_post_activation_monitor_evidence",
                    return_value={
                        "status": "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED"
                    },
                ),
                self.assertRaisesRegex(ValueError, "CURRENT_TRUTH_CHANGED"),
            ):
                monitor_tool._existing_valid_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=ledger_path,
                    record={"experiment_id": "experiment-monitor"},
                )

            with (
                patch.object(
                    monitor_tool,
                    "validate_post_activation_monitor_evidence",
                    return_value={"status": "MONITOR_EVIDENCE_CONTENT_ADDRESS_INVALID"},
                ),
                self.assertRaisesRegex(ValueError, "CONTENT_ADDRESS_INVALID"),
            ):
                monitor_tool._existing_valid_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=ledger_path,
                    record={"experiment_id": "experiment-monitor"},
                )

    def test_content_addressed_monitor_evidence_binds_decision_and_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            queue_path.parent.mkdir(parents=True)
            record = {
                "override_key": f"{LANE}|forecast_confidence_floor",
                "work_order_id": "work-order-monitor",
                "experiment_id": "experiment-monitor",
                "activation_manifest_ref": "data/manifest.json#sha256=" + "1" * 64,
                "terminal_confirmation_sha256": "2" * 64,
                "lane_id": LANE,
                "pair": "EUR_USD",
                "method": "TREND_CONTINUATION",
                "parameter": "forecast_confidence_floor",
                "candidate_value": 0.70,
                "activated_at_utc": "2026-07-11T00:00:00+00:00",
                "activation_ledger_anchor": _activation_ledger_anchor(),
            }
            cohort = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_COHORT_COMPLETE",
                "cohort_id": "3" * 64,
                "lane_id": LANE,
                "pair": "EUR_USD",
                "method": "TREND_CONTINUATION",
                "activated_at_utc": record["activated_at_utc"],
                "activation_ledger_anchor": record["activation_ledger_anchor"],
                "sample_count": 20,
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": 1.25,
            }
            sealed = monitor.seal_post_activation_monitor_evidence(
                project_root=root,
                override_record=record,
                cohort=cohort,
                now=datetime.now(timezone.utc),
            )
            with patch.object(
                monitor,
                "validate_post_activation_monitor_cohort",
                return_value={
                    "status": "VALID",
                    "cohort_id": cohort["cohort_id"],
                    "primary_metric_value": cohort["primary_metric_value"],
                },
            ):
                validated = monitor.validate_post_activation_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=root / "data" / "execution_ledger.db",
                    evidence_ref=sealed["evidence_ref"],
                    expected_record=record,
                )

            self.assertEqual(sealed["payload"]["decision"], "KEEP")
            self.assertEqual(validated["status"], "VALID")
            current_truth_changed = {
                "status": "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
                "current_status": "WAITING_FOR_FIRST_20_RESOLUTIONS",
            }
            with patch.object(
                monitor,
                "validate_post_activation_monitor_cohort",
                return_value=current_truth_changed,
            ):
                stale = monitor.validate_post_activation_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=root / "data" / "execution_ledger.db",
                    evidence_ref=sealed["evidence_ref"],
                    expected_record=record,
                )
            self.assertEqual(stale["status"], "MONITOR_EVIDENCE_COHORT_INVALID")
            self.assertEqual(stale["cohort_validation"], current_truth_changed)
            evidence_path = (
                root
                / str(sealed["evidence_ref"]).split("#", 1)[0]
            )
            evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
            invalid = monitor.validate_post_activation_monitor_evidence(
                queue_path=queue_path,
                ledger_path=root / "data" / "execution_ledger.db",
                evidence_ref=sealed["evidence_ref"],
                expected_record=record,
            )
            self.assertEqual(
                invalid["status"],
                "MONITOR_EVIDENCE_CONTENT_ADDRESS_INVALID",
            )

    def test_hand_authored_positive_outer_metric_cannot_hide_negative_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            queue_path.parent.mkdir(parents=True)
            record = {
                "override_key": f"{LANE}|forecast_confidence_floor",
                "work_order_id": "work-order-monitor",
                "experiment_id": "experiment-monitor",
                "activation_manifest_ref": "data/manifest.json#sha256=" + "1" * 64,
                "terminal_confirmation_sha256": "2" * 64,
                "lane_id": LANE,
                "pair": "EUR_USD",
                "method": "TREND_CONTINUATION",
                "parameter": "forecast_confidence_floor",
                "candidate_value": 0.70,
                "activated_at_utc": "2026-07-11T00:00:00+00:00",
                "activation_ledger_anchor": _activation_ledger_anchor(),
            }
            cohort = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_COHORT_COMPLETE",
                "cohort_id": "4" * 64,
                "lane_id": LANE,
                "pair": "EUR_USD",
                "method": "TREND_CONTINUATION",
                "activated_at_utc": record["activated_at_utc"],
                "activation_ledger_anchor": record["activation_ledger_anchor"],
                "sample_count": 20,
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": -2.0,
            }
            payload = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_MONITOR_COMPLETED",
                "decision": "KEEP",
                **record,
                "cohort_id": cohort["cohort_id"],
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": 2.0,
                "sample_count": 20,
                "cohort": cohort,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_permission_allowed": False,
                "no_direct_oanda": True,
            }
            evidence_ref = _write_hand_authored_monitor_evidence(root, payload)

            with patch.object(
                monitor,
                "validate_post_activation_monitor_cohort",
                return_value={
                    "status": "VALID",
                    "cohort_id": cohort["cohort_id"],
                    "primary_metric_value": -2.0,
                },
            ):
                result = monitor.validate_post_activation_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=root / "data" / "execution_ledger.db",
                    evidence_ref=evidence_ref,
                    expected_record=record,
                )

            self.assertEqual(
                result["status"],
                "MONITOR_EVIDENCE_COHORT_BINDING_INVALID",
            )

    def test_hand_authored_cohort_cannot_borrow_another_activation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            queue_path.parent.mkdir(parents=True)
            record = {
                "override_key": f"{LANE}|forecast_confidence_floor",
                "work_order_id": "work-order-monitor",
                "experiment_id": "experiment-monitor",
                "activation_manifest_ref": "data/manifest.json#sha256=" + "1" * 64,
                "terminal_confirmation_sha256": "2" * 64,
                "lane_id": LANE,
                "pair": "EUR_USD",
                "method": "TREND_CONTINUATION",
                "parameter": "forecast_confidence_floor",
                "candidate_value": 0.70,
                "activated_at_utc": "2026-07-11T00:00:00+00:00",
                "activation_ledger_anchor": _activation_ledger_anchor(),
            }
            borrowed_lane = "range_trader:USD_JPY:SHORT:RANGE_ROTATION:MARKET"
            cohort = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_COHORT_COMPLETE",
                "cohort_id": "5" * 64,
                "lane_id": borrowed_lane,
                "pair": "USD_JPY",
                "method": "RANGE_ROTATION",
                "activated_at_utc": "2026-07-10T00:00:00+00:00",
                "activation_ledger_anchor": {
                    **_activation_ledger_anchor(),
                    "ledger_prefix_sha256": "b" * 64,
                    "captured_at_utc": "2026-07-10T00:00:00+00:00",
                },
                "sample_count": 20,
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": 3.0,
            }
            payload = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_MONITOR_COMPLETED",
                "decision": "KEEP",
                **record,
                "cohort_id": cohort["cohort_id"],
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": 3.0,
                "sample_count": 20,
                "cohort": cohort,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_permission_allowed": False,
                "no_direct_oanda": True,
            }
            evidence_ref = _write_hand_authored_monitor_evidence(root, payload)

            with patch.object(
                monitor,
                "validate_post_activation_monitor_cohort",
                return_value={
                    "status": "VALID",
                    "cohort_id": cohort["cohort_id"],
                    "primary_metric_value": 3.0,
                },
            ):
                result = monitor.validate_post_activation_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=root / "data" / "execution_ledger.db",
                    evidence_ref=evidence_ref,
                    expected_record=record,
                )

            self.assertEqual(
                result["status"],
                "MONITOR_EVIDENCE_COHORT_BINDING_INVALID",
            )

    def test_hand_authored_cohort_cannot_replace_activation_ledger_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            queue_path.parent.mkdir(parents=True)
            record = {
                "override_key": f"{LANE}|forecast_confidence_floor",
                "work_order_id": "work-order-monitor",
                "experiment_id": "experiment-monitor",
                "activation_manifest_ref": "data/manifest.json#sha256=" + "1" * 64,
                "terminal_confirmation_sha256": "2" * 64,
                "lane_id": LANE,
                "pair": "EUR_USD",
                "method": "TREND_CONTINUATION",
                "parameter": "forecast_confidence_floor",
                "candidate_value": 0.70,
                "activated_at_utc": "2026-07-11T00:00:00+00:00",
                "activation_ledger_anchor": _activation_ledger_anchor(),
            }
            cohort = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_COHORT_COMPLETE",
                "cohort_id": "6" * 64,
                "lane_id": LANE,
                "pair": "EUR_USD",
                "method": "TREND_CONTINUATION",
                "activated_at_utc": record["activated_at_utc"],
                "activation_ledger_anchor": {
                    **_activation_ledger_anchor(),
                    "ledger_prefix_sha256": "b" * 64,
                },
                "sample_count": 20,
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": 3.0,
            }
            payload = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_MONITOR_COMPLETED",
                "decision": "KEEP",
                **record,
                "cohort_id": cohort["cohort_id"],
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_value": 3.0,
                "sample_count": 20,
                "cohort": cohort,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_permission_allowed": False,
                "no_direct_oanda": True,
            }
            evidence_ref = _write_hand_authored_monitor_evidence(root, payload)

            with patch.object(
                monitor,
                "validate_post_activation_monitor_cohort",
                return_value={
                    "status": "VALID",
                    "cohort_id": cohort["cohort_id"],
                    "primary_metric_value": 3.0,
                },
            ):
                result = monitor.validate_post_activation_monitor_evidence(
                    queue_path=queue_path,
                    ledger_path=root / "data" / "execution_ledger.db",
                    evidence_ref=evidence_ref,
                    expected_record=record,
                )

            self.assertEqual(
                result["status"],
                "MONITOR_EVIDENCE_COHORT_BINDING_INVALID",
            )

    @patch(
        "quant_rabbit.guardian_tuning_monitor.validate_post_activation_monitor_evidence",
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_quarantine_head_fails_closed_until_state_confirmation(
        self,
        _strict_queue: object,
        _strict_terminal: object,
        validate_monitor: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            override_path = root / "data" / "guardian_tuning_overrides.json"
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/a.json#sha256=" + "a" * 64
            staged = overrides.apply_accepted_override(
                path=override_path,
                work_order={"work_order_id": "work-order-monitor", "bot_tuning_review": _review()},
                prepared_contract=_contract(),
                experiment_id="experiment-monitor",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=_activation_ledger_anchor(),
                now=datetime.now(timezone.utc),
            )
            _write_terminal_queue(
                queue_path=queue_path,
                application=staged,
                experiment_id="experiment-monitor",
                evidence_ref=evidence_ref,
            )
            overrides.confirm_accepted_override(
                path=override_path,
                queue_path=queue_path,
                work_order_id="work-order-monitor",
                experiment_id="experiment-monitor",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )
            record = overrides.read_active_override_records(path=override_path)[0]
            monitor_ref = (
                "data/guardian_tuning_monitor_evidence/"
                + "b" * 64
                + ".json#sha256="
                + "b" * 64
            )
            validate_monitor.return_value = {
                "status": "VALID",
                "decision": "QUARANTINE",
                "primary_metric_value": -1.0,
            }
            queue = json.loads(queue_path.read_text())
            head = queue["override_lifecycle_heads"][0]
            head.update(
                {
                    "status": "QUARANTINED_COMMITTED",
                    "monitor_decision": "QUARANTINE",
                    "monitor_evidence_ref": monitor_ref,
                    "post_activation_primary_metric": -1.0,
                    "monitored_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            queue_path.write_text(json.dumps(queue))

            split = overrides.resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id=LANE,
                fallback=0.65,
                path=override_path,
                queue_path=queue_path,
            )
            self.assertEqual(split["status"], "OVERRIDE_STATE_INVALID")
            confirmed = overrides.confirm_post_activation_monitor(
                path=override_path,
                queue_path=queue_path,
                override_key=str(record["override_key"]),
                experiment_id="experiment-monitor",
                monitor_evidence_ref=monitor_ref,
                decision="QUARANTINE",
                primary_metric_value=-1.0,
                now=datetime.now(timezone.utc),
            )
            quarantined = overrides.resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id=LANE,
                fallback=0.65,
                path=override_path,
                queue_path=queue_path,
            )

            self.assertEqual(
                confirmed["status"],
                "POST_ACTIVATION_LANE_QUARANTINED",
            )
            self.assertEqual(quarantined["status"], "OVERRIDE_LANE_QUARANTINED")

    def test_dispatcher_commits_keep_head_and_confirms_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            override_path = root / "data" / "guardian_tuning_overrides.json"
            queue_path = root / "data" / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/c.json#sha256=" + "c" * 64
            strict_queue = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            with (
                patch.object(
                    overrides,
                    "_strict_queue_source_sha256",
                    side_effect=strict_queue,
                ),
                patch.object(
                    overrides,
                    "_strict_terminal_evidence_error",
                    return_value=None,
                ),
            ):
                staged = overrides.apply_accepted_override(
                    path=override_path,
                    work_order={
                        "work_order_id": "work-order-monitor",
                        "bot_tuning_review": _review(),
                    },
                    prepared_contract=_contract(),
                    experiment_id="experiment-keep",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref=evidence_ref,
                    activation_ledger_anchor=_activation_ledger_anchor(),
                    now=datetime.now(timezone.utc),
                )
                _write_terminal_queue(
                    queue_path=queue_path,
                    application=staged,
                    experiment_id="experiment-keep",
                    evidence_ref=evidence_ref,
                )
                overrides.confirm_accepted_override(
                    path=override_path,
                    queue_path=queue_path,
                    work_order_id="work-order-monitor",
                    experiment_id="experiment-keep",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref=evidence_ref,
                    now=datetime.now(timezone.utc),
                )
                record = overrides.read_active_override_records(path=override_path)[0]
                monitor_ref = (
                    "data/guardian_tuning_monitor_evidence/"
                    + "d" * 64
                    + ".json#sha256="
                    + "d" * 64
                )

                def load_queue(path: Path) -> dict:
                    raw = path.read_bytes()
                    payload = json.loads(raw)
                    payload["_queue_source_sha256"] = hashlib.sha256(raw).hexdigest()
                    return payload

                valid_monitor = {
                    "status": "VALID",
                    "decision": "KEEP",
                    "primary_metric_value": 2.0,
                }
                with (
                    patch.object(
                        dispatcher,
                        "_load_tuning_work_order",
                        side_effect=load_queue,
                    ),
                    patch.object(
                        dispatcher,
                        "validate_post_activation_monitor_evidence",
                        return_value=valid_monitor,
                    ),
                    patch.object(
                        monitor,
                        "validate_post_activation_monitor_evidence",
                        return_value=valid_monitor,
                    ),
                ):
                    result = dispatcher.commit_tuning_override_monitor(
                        path=queue_path,
                        override_path=override_path,
                        override_key=str(record["override_key"]),
                        experiment_id="experiment-keep",
                        monitor_evidence_ref=monitor_ref,
                        decision="KEEP",
                        primary_metric_value=2.0,
                        now=datetime.now(timezone.utc),
                    )
                    resolution = overrides.resolve_forecast_confidence_floor_state(
                        pair="EUR_USD",
                        method="TREND_CONTINUATION",
                        lane_id=LANE,
                        fallback=0.65,
                        path=override_path,
                        queue_path=queue_path,
                    )

            self.assertEqual(result["status"], "POST_ACTIVATION_MONITOR_COMMITTED")
            self.assertEqual(
                json.loads(queue_path.read_text())["override_lifecycle_heads"][0]["status"],
                "MONITORED_KEEP_COMMITTED",
            )
            self.assertEqual(resolution["status"], "ACTIVE_OVERRIDE")
            self.assertEqual(resolution["override"]["monitor_decision"], "KEEP")


if __name__ == "__main__":
    unittest.main()

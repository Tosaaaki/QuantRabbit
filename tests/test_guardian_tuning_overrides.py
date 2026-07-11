from __future__ import annotations

import asyncio
import contextvars
import fcntl
import hashlib
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit import guardian_tuning_overrides as override_module
from quant_rabbit.guardian_tuning_overrides import (
    apply_accepted_override,
    confirm_accepted_override,
    reconcile_pending_overrides,
    resolve_forecast_confidence_floor,
    resolve_forecast_confidence_floor_state,
)


class GuardianTuningOverridesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._monitor_gate = patch(
            "quant_rabbit.guardian_tuning_cohort.build_post_activation_monitor_cohort",
            return_value={
                "schema_version": 1,
                "status": "WAITING_FOR_FIRST_20_ENTRIES",
                "entry_count": 0,
                "required_entry_count": 20,
            },
        )
        self._monitor_gate_mock = self._monitor_gate.start()
        self.addCleanup(self._monitor_gate.stop)

    def _work_order(self, *, current: float = 0.65, candidate: float = 0.70) -> dict:
        return {
            "work_order_id": "work-order-1",
            "bot_tuning_review": {
                "review_status": "TEST_REQUIRED",
                "proposed_adjustments": [
                    {
                        "pair": "EUR_USD",
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                        "bot_family": "forecast",
                        "parameter": "forecast_confidence_floor",
                        "current_value": current,
                        "candidate_value": candidate,
                    }
                ],
            }
        }

    def test_runtime_validation_cache_is_scoped_and_explicitly_refreshable(self) -> None:
        result = {"status": "NO_OVERRIDE", "resolved_value": 0.65}
        call = {
            "pair": "EUR_USD",
            "method": "TREND_CONTINUATION",
            "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
            "fallback": 0.65,
        }
        with patch.object(
            override_module,
            "_resolve_forecast_confidence_floor_state_uncached",
            return_value=result,
        ) as uncached:
            resolve_forecast_confidence_floor_state(**call)
            resolve_forecast_confidence_floor_state(**call)
            self.assertEqual(uncached.call_count, 2)

            with override_module.guardian_tuning_validation_cycle():
                resolve_forecast_confidence_floor_state(**call)
                resolve_forecast_confidence_floor_state(**call)
                self.assertEqual(uncached.call_count, 3)
                override_module.clear_guardian_tuning_validation_cache()
                resolve_forecast_confidence_floor_state(**call)
                self.assertEqual(uncached.call_count, 4)

            resolve_forecast_confidence_floor_state(**call)
            self.assertEqual(uncached.call_count, 5)

    def test_cache_generation_rejects_inflight_write_after_clear(self) -> None:
        call = {
            "pair": "EUR_USD",
            "method": "TREND_CONTINUATION",
            "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
            "fallback": 0.65,
        }
        started = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()
        child_results: list[dict] = []

        def resolve_uncached(**_kwargs: object) -> dict:
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                started.set()
                self.assertTrue(release.wait(timeout=5.0))
                return {"status": "ACTIVE_OVERRIDE", "resolved_value": 0.70}
            return {
                "status": "OVERRIDE_LANE_QUARANTINED",
                "resolved_value": 0.70,
            }

        with patch.object(
            override_module,
            "_resolve_forecast_confidence_floor_state_uncached",
            side_effect=resolve_uncached,
        ):
            with override_module.guardian_tuning_validation_cycle():
                child_context = contextvars.copy_context()
                worker = threading.Thread(
                    target=lambda: child_results.append(
                        child_context.run(
                            resolve_forecast_confidence_floor_state,
                            **call,
                        )
                    )
                )
                worker.start()
                self.assertTrue(started.wait(timeout=5.0))
                override_module.clear_guardian_tuning_validation_cache()
                release.set()
                worker.join(timeout=5.0)
                self.assertFalse(worker.is_alive())

                final = resolve_forecast_confidence_floor_state(**call)

        self.assertEqual(child_results[0]["status"], "ACTIVE_OVERRIDE")
        self.assertEqual(final["status"], "OVERRIDE_LANE_QUARANTINED")
        self.assertEqual(calls, 2)

    def test_owner_exit_revokes_copied_context_and_nested_phase_is_independent(
        self,
    ) -> None:
        call = {
            "pair": "EUR_USD",
            "method": "TREND_CONTINUATION",
            "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
            "fallback": 0.65,
        }
        result = {"status": "NO_OVERRIDE", "resolved_value": 0.65}
        with patch.object(
            override_module,
            "_resolve_forecast_confidence_floor_state_uncached",
            return_value=result,
        ) as uncached:
            with override_module.guardian_tuning_validation_cycle():
                resolve_forecast_confidence_floor_state(**call)
                child_context = contextvars.copy_context()
                with override_module.guardian_tuning_validation_cycle():
                    resolve_forecast_confidence_floor_state(**call)
                resolve_forecast_confidence_floor_state(**call)
                self.assertEqual(uncached.call_count, 2)

            child_context.run(resolve_forecast_confidence_floor_state, **call)
            child_context.run(resolve_forecast_confidence_floor_state, **call)

        # The copied child sees the closed cycle object, so neither call may
        # hit or repopulate the cache that belonged to the exited owner.
        self.assertEqual(uncached.call_count, 4)

    def test_async_decorator_keeps_one_owned_phase_across_await(self) -> None:
        call = {
            "pair": "EUR_USD",
            "method": "TREND_CONTINUATION",
            "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
            "fallback": 0.65,
        }
        result = {"status": "NO_OVERRIDE", "resolved_value": 0.65}

        @override_module.guardian_tuning_validation_cycle()
        async def validate_twice() -> None:
            resolve_forecast_confidence_floor_state(**call)
            await asyncio.sleep(0)
            resolve_forecast_confidence_floor_state(**call)

        with patch.object(
            override_module,
            "_resolve_forecast_confidence_floor_state_uncached",
            return_value=result,
        ) as uncached:
            asyncio.run(validate_twice())
            self.assertEqual(uncached.call_count, 1)
            resolve_forecast_confidence_floor_state(**call)
            self.assertEqual(uncached.call_count, 2)

    @staticmethod
    def _contract() -> dict:
        return {
            "experiment_contract_digest": "a" * 64,
            "source_identity": {
                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
            },
            "active_parameter_binding": {
                "parameter": "forecast_confidence_floor",
                "environment_variable": "QR_FORECAST_DIRECTIONAL_LIVE_MIN_CONFIDENCE",
                "resolved_value": 0.65,
                "method": "TREND_CONTINUATION",
            },
        }

    @staticmethod
    def _activation_anchor() -> dict:
        return {
            "ledger_rowid_watermark": 1,
            "ledger_prefix_sha256": "a" * 64,
            "execution_ledger_coverage_start_utc": "2026-01-01T00:00:00+00:00",
            "last_oanda_transaction_id": "1",
            "captured_at_utc": "2026-07-11T00:00:00+00:00",
        }

    def _write_terminal_queue(
        self,
        *,
        path: Path,
        application: dict,
        experiment_id: str,
        evidence_ref: str,
    ) -> None:
        terminal = {
            "work_order_id": "work-order-1",
            "status": "CONSUMED",
            "experiment_id": experiment_id,
            "experiment_result": "ACCEPTED_IMPROVEMENT",
            "experiment_evidence_ref": evidence_ref,
            "experiment_contract_digest": "a" * 64,
            "terminal_transition_source": "guardian_tuning_work_order_lifecycle",
            "tuning_override_application": application,
            "bot_tuning_review": self._work_order()["bot_tuning_review"],
            "prepared_experiment_contract": self._contract(),
            "live_permission_allowed": False,
            "no_direct_oanda": True,
        }
        terminal_digest = hashlib.sha256(
            json.dumps(
                terminal,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        terminal_record_ref = override_module.write_terminal_commitment_manifest(
            queue_path=path,
            terminal=terminal,
        )
        head = {
            "status": "ACTIVE_COMMITTED",
            "override_key": application["override_key"],
            "work_order_id": "work-order-1",
            "experiment_id": experiment_id,
            "experiment_result": "ACCEPTED_IMPROVEMENT",
            "experiment_evidence_ref": evidence_ref,
            "experiment_contract_digest": "a" * 64,
            "terminal_confirmation_sha256": terminal_digest,
            "terminal_record_ref": terminal_record_ref,
            "pair": "EUR_USD",
            "method": "TREND_CONTINUATION",
            "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
            "parameter": "forecast_confidence_floor",
            "candidate_value": application["candidate_value"],
            "activated_at_utc": application["activated_at_utc"],
            "activation_ledger_anchor": application["activation_ledger_anchor"],
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
            "live_permission_allowed": False,
            "no_direct_oanda": True,
        }
        payload = {
            **terminal,
            "schema_version": 2,
            "queue_schema_revision": 4,
            "work_orders": [],
            "pending_count": 0,
            "terminal_history": [terminal],
            "terminal_history_count": 1,
            "experiment_semantic_digest_history": [],
            "experiment_id_digest_history": [
                hashlib.sha256(
                    (
                        "guardian-tuning-experiment-id-v1\0" + experiment_id
                    ).encode("utf-8")
                ).hexdigest()
            ],
            "override_lifecycle_heads": [head],
        }
        path.write_text(json.dumps(payload))

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_accepted_forward_evidence_activates_idempotent_tightening_and_rollback_value(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/a.json#sha256=" + "a" * 64
            first = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-1",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            before_commit = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
            )
            with self.assertRaisesRegex(ValueError, "terminal tuning queue is missing"):
                confirm_accepted_override(
                    path=path,
                    queue_path=queue_path,
                    work_order_id="work-order-1",
                    experiment_id="experiment-1",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref=evidence_ref,
                    now=datetime.now(timezone.utc),
                )
            self._write_terminal_queue(
                path=queue_path,
                application=first,
                experiment_id="experiment-1",
                evidence_ref=evidence_ref,
            )
            confirmed = confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id="work-order-1",
                experiment_id="experiment-1",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )
            repeated = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-1",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )

            self.assertEqual(first["status"], "OVERRIDE_STAGED")
            self.assertEqual(first["rollback_value"], 0.65)
            self.assertEqual(
                before_commit["status"],
                "OVERRIDE_CONFIRMATION_PENDING",
            )
            self.assertEqual(confirmed["status"], "OVERRIDE_ACTIVATED")
            self.assertEqual(repeated["status"], "OVERRIDE_ALREADY_ACTIVE")
            self.assertEqual(
                resolve_forecast_confidence_floor(
                    pair="EUR_USD",
                    method="TREND_CONTINUATION",
                    lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                    fallback=0.65,
                    path=path,
                    queue_path=queue_path,
                ),
                0.70,
            )
            other_lane = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
            )
            missing_lane = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                fallback=0.65,
                path=path,
            )
            self.assertEqual(other_lane["status"], "NO_OVERRIDE")
            self.assertEqual(missing_lane["status"], "OVERRIDE_LANE_ID_REQUIRED")

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_active_state_requires_manifest_and_durable_queue_head(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/proof.json#sha256=" + "d" * 64
            staged = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-provenance",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=staged,
                experiment_id="experiment-provenance",
                evidence_ref=evidence_ref,
            )
            confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id="work-order-1",
                experiment_id="experiment-provenance",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )
            active = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(active["status"], "ACTIVE_OVERRIDE")
            case_variant = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:eur_usd:long:trend_continuation:limit",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(case_variant["status"], "OVERRIDE_LANE_ID_INVALID")

            committed_queue = queue_path.read_bytes()
            without_head = json.loads(committed_queue)
            without_head["override_lifecycle_heads"] = []
            queue_path.write_text(json.dumps(without_head))
            uncommitted = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(uncommitted["status"], "OVERRIDE_STATE_INVALID")

            queue_path.write_bytes(committed_queue)
            override_module._write(
                path,
                {
                    "schema_version": 1,
                    "active_overrides": [],
                    "pending_overrides": [],
                    "history": [],
                },
            )
            valid_empty_state = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(
                valid_empty_state["status"],
                "OVERRIDE_STATE_MISSING_WITH_COMMITMENT",
            )

            path.unlink()
            missing_state = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(
                missing_state["status"],
                "OVERRIDE_STATE_MISSING_WITH_COMMITMENT",
            )

            for manifest in (
                path.parent / "guardian_tuning_activation_manifests"
            ).glob("*.json"):
                manifest.unlink()
            queue_only = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(
                queue_only["status"],
                "OVERRIDE_STATE_MISSING_WITH_COMMITMENT",
            )

    def test_minimal_handwritten_queue_cannot_confirm_without_strict_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/missing.json#sha256=" + "e" * 64
            staged = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-handwritten",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=staged,
                experiment_id="experiment-handwritten",
                evidence_ref=evidence_ref,
            )

            with self.assertRaisesRegex(ValueError, "strict tuning queue validation failed"):
                confirm_accepted_override(
                    path=path,
                    queue_path=queue_path,
                    work_order_id="work-order-1",
                    experiment_id="experiment-handwritten",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref=evidence_ref,
                    now=datetime.now(timezone.utc),
                )

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_terminal_manifest_recovers_confirmation_after_history_trim(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/trim.json#sha256=" + "f" * 64
            staged = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-trimmed",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=staged,
                experiment_id="experiment-trimmed",
                evidence_ref=evidence_ref,
            )
            queue = json.loads(queue_path.read_text())
            queue["terminal_history"] = []
            queue["terminal_history_count"] = 0
            queue_path.write_text(json.dumps(queue))

            confirmed = confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id="work-order-1",
                experiment_id="experiment-trimmed",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )

            self.assertEqual(confirmed["status"], "OVERRIDE_ACTIVATED")
            active = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(active["status"], "ACTIVE_OVERRIDE")

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_active_candidate_cannot_diverge_from_committed_terminal(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/tamper.json#sha256=" + "1" * 64
            staged = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-candidate-tamper",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=staged,
                experiment_id="experiment-candidate-tamper",
                evidence_ref=evidence_ref,
            )
            confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id="work-order-1",
                experiment_id="experiment-candidate-tamper",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )

            state = json.loads(path.read_text())
            state.pop("state_digest_sha256", None)
            record = state["active_overrides"][0]
            manifest_ref = str(record["activation_manifest_ref"])
            manifest_name = manifest_ref.split("/", 2)[-1].split("#", 1)[0]
            manifest_path = path.parent / "guardian_tuning_activation_manifests" / manifest_name
            manifest = json.loads(manifest_path.read_text())
            manifest["candidate_value"] = 0.66
            manifest_raw = (
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
            (manifest_path.parent / f"{manifest_digest}.json").write_bytes(manifest_raw)
            record["candidate_value"] = 0.66
            record["activation_manifest_ref"] = (
                "data/guardian_tuning_activation_manifests/"
                f"{manifest_digest}.json#sha256={manifest_digest}"
            )
            override_module._write(path, state)
            queue = json.loads(queue_path.read_text())
            queue["override_lifecycle_heads"][0]["candidate_value"] = 0.66
            queue_path.write_text(json.dumps(queue))

            resolution = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )

            self.assertEqual(resolution["status"], "OVERRIDE_STATE_INVALID")

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_active_terminal_history_tamper_is_rejected(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/original.json#sha256=" + "2" * 64
            staged = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-terminal-tamper",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=staged,
                experiment_id="experiment-terminal-tamper",
                evidence_ref=evidence_ref,
            )
            confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id="work-order-1",
                experiment_id="experiment-terminal-tamper",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )
            queue = json.loads(queue_path.read_text())
            queue["terminal_history"][0]["experiment_evidence_ref"] = "tampered"
            queue_path.write_text(json.dumps(queue))

            resolution = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )

            self.assertEqual(resolution["status"], "OVERRIDE_STATE_INVALID")

    def test_corrupt_override_state_is_distinct_from_no_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            missing = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
            )
            path.write_text("{not-json")
            corrupt = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
            )

            self.assertEqual(missing["status"], "NO_OVERRIDE")
            self.assertEqual(corrupt["status"], "OVERRIDE_STATE_INVALID")

    def test_legacy_four_part_lane_is_advisory_without_any_override_commitment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            queue_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "work_orders": [],
                        "pending_count": 0,
                        "terminal_history": [],
                        "terminal_history_count": 0,
                    }
                )
            )
            resolution = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )

            self.assertEqual(resolution["status"], "NO_OVERRIDE")

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_reconciler_activates_only_a_matching_terminal_stage(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/r.json#sha256=" + "b" * 64
            staged = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-reconcile",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            unresolved = reconcile_pending_overrides(
                path=path,
                queue_path=queue_path,
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=staged,
                experiment_id="experiment-reconcile",
                evidence_ref=evidence_ref,
            )
            recovered = reconcile_pending_overrides(
                path=path,
                queue_path=queue_path,
                now=datetime.now(timezone.utc),
            )

            self.assertEqual(unresolved["status"], "OVERRIDE_CONFIRMATION_PENDING")
            self.assertEqual(recovered["status"], "PENDING_OVERRIDES_RECONCILED")
            self.assertEqual(recovered["reconciled_count"], 1)

    def test_reconciler_uses_the_queue_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            lock_path = queue_path.with_name(f"{queue_path.name}.lock")
            with lock_path.open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = reconcile_pending_overrides(
                    path=path,
                    queue_path=queue_path,
                    now=datetime.now(timezone.utc),
                )

            self.assertEqual(
                result["status"],
                "OVERRIDE_RECONCILIATION_CONCURRENT_UPDATE",
            )

    def test_rejected_result_does_not_create_runtime_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            result = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-rejected",
                experiment_result="REJECTED_NO_IMPROVEMENT",
                evidence_ref="data/guardian_tuning_evidence/b.json#sha256=" + "b" * 64,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )

            self.assertFalse(result["applied"])
            self.assertFalse(path.exists())

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_first_20_monitor_becomes_a_mandatory_runtime_gate(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/gate.json#sha256=" + "9" * 64
            staged = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-monitor-gate",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=staged,
                experiment_id="experiment-monitor-gate",
                evidence_ref=evidence_ref,
            )
            confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id="work-order-1",
                experiment_id="experiment-monitor-gate",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )

            before_20 = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(before_20["status"], "ACTIVE_OVERRIDE")

            self._monitor_gate_mock.return_value = {
                "schema_version": 1,
                "status": "WAITING_FOR_FIRST_20_RESOLUTIONS",
                "entry_count": 20,
                "resolved_count": 19,
                "required_resolved_count": 20,
            }
            unresolved = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(
                unresolved["status"],
                "OVERRIDE_POST_ACTIVATION_MONITOR_PENDING",
            )
            with self.assertRaisesRegex(ValueError, "MONITOR_PENDING"):
                override_module.runtime_forecast_floor_binding(
                    lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                    override_path=path,
                    queue_path=queue_path,
                )
            monitor_binding = override_module.runtime_forecast_floor_binding(
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                override_path=path,
                queue_path=queue_path,
                allow_post_activation_monitor_pending=True,
            )
            self.assertEqual(monitor_binding["resolved_value"], 0.70)

            self._monitor_gate_mock.return_value = {
                "schema_version": 1,
                "status": "POST_ACTIVATION_COHORT_COMPLETE",
                "sample_count": 20,
            }
            complete_uncommitted = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(
                complete_uncommitted["status"],
                "OVERRIDE_POST_ACTIVATION_MONITOR_PENDING",
            )

            self._monitor_gate_mock.return_value = {
                "schema_version": 1,
                "status": "WAITING_FOR_FIRST_20_ENTRIES",
                "entry_count": 20,
                "required_entry_count": 20,
            }
            malformed = resolve_forecast_confidence_floor_state(
                pair="EUR_USD",
                method="TREND_CONTINUATION",
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT",
                fallback=0.65,
                path=path,
                queue_path=queue_path,
            )
            self.assertEqual(malformed["status"], "OVERRIDE_STATE_INVALID")

    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_terminal_evidence_error",
        return_value=None,
    )
    @patch(
        "quant_rabbit.guardian_tuning_overrides._strict_queue_source_sha256",
        side_effect=lambda queue_path: hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    )
    def test_successor_requires_prior_monitor_keep(
        self,
        _strict_queue: object,
        _strict_evidence: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            queue_path = Path(tmp) / "guardian_tuning_work_order.json"
            evidence_ref = "data/guardian_tuning_evidence/prior.json#sha256=" + "8" * 64
            first = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-prior",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            self._write_terminal_queue(
                path=queue_path,
                application=first,
                experiment_id="experiment-prior",
                evidence_ref=evidence_ref,
            )
            confirm_accepted_override(
                path=path,
                queue_path=queue_path,
                work_order_id="work-order-1",
                experiment_id="experiment-prior",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref=evidence_ref,
                now=datetime.now(timezone.utc),
            )
            next_contract = self._contract()
            next_contract["active_parameter_binding"]["resolved_value"] = 0.70
            next_work_order = self._work_order(current=0.70, candidate=0.75)
            with self.assertRaisesRegex(ValueError, "monitor was not kept"):
                apply_accepted_override(
                    path=path,
                    work_order=next_work_order,
                    prepared_contract=next_contract,
                    experiment_id="experiment-too-early",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref="data/guardian_tuning_evidence/early.json#sha256="
                    + "7" * 64,
                    activation_ledger_anchor=self._activation_anchor(),
                    now=datetime.now(timezone.utc),
                )

            state = override_module._load(path)
            record = dict(state["active_overrides"][0])
            record.update(
                {
                    "activation_status": "ACTIVE",
                    "monitor_decision": "KEEP",
                    "monitor_evidence_ref": (
                        "data/guardian_tuning_monitor_evidence/"
                        + "6" * 64
                        + ".json#sha256="
                        + "6" * 64
                    ),
                    "post_activation_primary_metric": 1.0,
                    "monitored_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            override_module._write(
                path,
                {
                    "schema_version": 1,
                    "active_overrides": [record],
                    "pending_overrides": [],
                    "history": [],
                },
            )
            queue = json.loads(queue_path.read_text())
            queue_head = queue["override_lifecycle_heads"][0]
            queue_head.update(
                {
                    "status": "MONITORED_KEEP_COMMITTED",
                    "monitor_decision": "KEEP",
                    "monitor_evidence_ref": record["monitor_evidence_ref"],
                    "post_activation_primary_metric": 1.0,
                    "monitored_at_utc": record["monitored_at_utc"],
                }
            )
            queue_path.write_text(json.dumps(queue))
            with self.assertRaisesRegex(ValueError, "current provenance is invalid"):
                apply_accepted_override(
                    path=path,
                    work_order=next_work_order,
                    prepared_contract=next_contract,
                    experiment_id="experiment-missing-prior-proof",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref=(
                        "data/guardian_tuning_evidence/missing-prior.json#sha256="
                        + "0" * 64
                    ),
                    activation_ledger_anchor=self._activation_anchor(),
                    now=datetime.now(timezone.utc),
                    queue_path=queue_path,
                )
            with patch(
                "quant_rabbit.guardian_tuning_monitor."
                "validate_post_activation_monitor_evidence",
                return_value={
                    "status": "VALID",
                    "decision": "KEEP",
                    "primary_metric_value": 1.0,
                },
            ):
                allowed = apply_accepted_override(
                    path=path,
                    work_order=next_work_order,
                    prepared_contract=next_contract,
                    experiment_id="experiment-after-keep",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref="data/guardian_tuning_evidence/kept.json#sha256="
                    + "5" * 64,
                    activation_ledger_anchor=self._activation_anchor(),
                    now=datetime.now(timezone.utc),
                    queue_path=queue_path,
                )
                predecessor = (
                    override_module.read_validated_kept_predecessor_record(
                        path=path,
                        queue_path=queue_path,
                        override_key=str(record["override_key"]),
                        experiment_id="experiment-prior",
                    )
                )
            self.assertEqual(allowed["status"], "OVERRIDE_STAGED")
            self.assertEqual(predecessor["experiment_id"], "experiment-prior")
            with (
                patch(
                    "quant_rabbit.guardian_tuning_monitor."
                    "validate_post_activation_monitor_evidence",
                    return_value={"status": "MONITOR_EVIDENCE_READ_FAILED"},
                ),
                self.assertRaisesRegex(ValueError, "monitor evidence is invalid"),
            ):
                override_module.read_validated_kept_predecessor_record(
                    path=path,
                    queue_path=queue_path,
                    override_key=str(record["override_key"]),
                    experiment_id="experiment-prior",
                )

    def test_quarantined_override_cannot_be_replaced_by_successor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_tuning_overrides.json"
            first = apply_accepted_override(
                path=path,
                work_order=self._work_order(),
                prepared_contract=self._contract(),
                experiment_id="experiment-quarantined",
                experiment_result="ACCEPTED_IMPROVEMENT",
                evidence_ref="data/guardian_tuning_evidence/q.json#sha256=" + "4" * 64,
                activation_ledger_anchor=self._activation_anchor(),
                now=datetime.now(timezone.utc),
            )
            state = override_module._load(path)
            pending = dict(state["pending_overrides"][0])
            pending.update(
                {
                    "activation_status": "QUARANTINED",
                    "monitor_decision": "QUARANTINE",
                }
            )
            override_module._write(
                path,
                {
                    "schema_version": 1,
                    "active_overrides": [pending],
                    "pending_overrides": [],
                    "history": [],
                },
            )
            next_contract = self._contract()
            next_contract["active_parameter_binding"]["resolved_value"] = 0.70
            with self.assertRaisesRegex(ValueError, "monitor was not kept"):
                apply_accepted_override(
                    path=path,
                    work_order=self._work_order(current=0.70, candidate=0.75),
                    prepared_contract=next_contract,
                    experiment_id="experiment-after-quarantine",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref="data/guardian_tuning_evidence/q2.json#sha256="
                    + "3" * 64,
                    activation_ledger_anchor=self._activation_anchor(),
                    now=datetime.now(timezone.utc),
                )

    def test_reviewed_current_must_still_match_runtime_before_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "current value"):
                apply_accepted_override(
                    path=Path(tmp) / "guardian_tuning_overrides.json",
                    work_order=self._work_order(current=0.60, candidate=0.70),
                    prepared_contract=self._contract(),
                    experiment_id="experiment-stale-current",
                    experiment_result="ACCEPTED_IMPROVEMENT",
                    evidence_ref="data/guardian_tuning_evidence/c.json#sha256=" + "c" * 64,
                    activation_ledger_anchor=self._activation_anchor(),
                    now=datetime.now(timezone.utc),
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.dojo_ai_discretion import (
    CAPABILITY_MANIFEST_CONTRACT,
    DIAGNOSTIC_TIER,
    FIXED_RESPONSE_KEYS,
    PACKET_CONTRACT,
    PROMPT_LOCK_CONTRACT,
    RESPONSE_CONTRACT,
    SCORE_CONTRACT,
    SCORER_LOCK_CONTRACT,
    SOURCE_CONTRACT,
    assert_no_stale_positive_artifacts,
    build_capability_manifest,
    build_day_packet,
    canonical_sha256,
    invalidate_artifact,
    prelock_prompt,
    prelock_scorer,
    score_pilot,
    score_sealed_response,
    seal_answer_key,
    seal_model_manifest,
    seal_response,
    seal_validity_registry,
)


NOW = datetime(2026, 7, 19, 7, 0, tzinfo=timezone.utc)


def _source(**updates: object) -> dict[str, object]:
    source: dict[str, object] = {
        "contract": SOURCE_CONTRACT,
        "blind_nonce": "f" * 64,
        "pair": "USD_JPY",
        "decision_cutoff_utc": NOW.isoformat(),
        "observations": [
            {
                "observed_at_utc": (NOW - timedelta(hours=2)).isoformat(),
                "kind": "ASIAN_SESSION",
                "payload": {"open": 156.2, "high": 156.7, "low": 156.1, "last": 156.5},
            },
            {
                "observed_at_utc": (NOW - timedelta(minutes=5)).isoformat(),
                "kind": "CLOSED_BAR",
                "payload": {"timeframe": "M5", "open": 156.4, "close": 156.5},
            },
        ],
    }
    source.update(updates)
    return source


def _locked_artifacts() -> tuple[dict[str, object], ...]:
    capability = build_capability_manifest(
        context_id="fresh-context-001",
        generated_at_utc=NOW - timedelta(minutes=3),
    )
    prompt = prelock_prompt(
        "Use only the inline packet. Return the fixed JSON decision schema.",
        variant_id="neutral-v1",
        locked_at_utc=NOW - timedelta(minutes=2),
    )
    scorer = prelock_scorer(locked_at_utc=NOW - timedelta(minutes=2))
    model = seal_model_manifest(
        model_name="offline-test-model",
        model_version="2026-07-19",
        model_lineage="lineage-a",
        reasoning_effort="high",
        context_id="fresh-context-001",
        capability_manifest=capability,
        locked_at_utc=NOW - timedelta(minutes=1),
    )
    return capability, prompt, scorer, model


def _packet() -> tuple[dict[str, object], ...]:
    capability, prompt, scorer, model = _locked_artifacts()
    packet = build_day_packet(
        _source(),
        prompt_lock=prompt,
        capability_manifest=capability,
        scorer_lock=scorer,
    )
    return packet, capability, prompt, scorer, model


def _decision(packet: dict[str, object], **updates: object) -> dict[str, object]:
    decision: dict[str, object] = {
        "trial_id": packet["trial_id"],
        "action": "LONG",
        "pair": "USD_JPY",
        "size": "HALF",
        "confidence": 0.62,
        "evidence_refs": ["obs-001", "obs-002"],
        "target_pips": 35.0,
        "invalidation_pips": 18.0,
        "strongest_counterargument": "The Asian move may already be exhausted.",
        "abstain_reason": None,
    }
    decision.update(updates)
    return decision


def _sealed_response(*, lineage: str = "lineage-a") -> tuple[dict[str, object], ...]:
    return _sealed_response_with_parents(lineage=lineage)


def _sealed_response_with_parents(
    *, lineage: str = "lineage-a"
) -> tuple[dict[str, object], ...]:
    packet, capability, prompt, scorer, model = _packet()
    if lineage != "lineage-a":
        model = seal_model_manifest(
            model_name="offline-test-model",
            model_version="2026-07-19",
            model_lineage=lineage,
            reasoning_effort="high",
            context_id="fresh-context-001",
            capability_manifest=capability,
            locked_at_utc=NOW - timedelta(minutes=1),
        )
    response = seal_response(
        _decision(packet),
        packet=packet,
        prompt_lock=prompt,
        model_manifest=model,
        capability_manifest=capability,
        sealed_at_utc=NOW,
    )
    return response, packet, capability, prompt, scorer, model


class DojoAiDiscretionTest(unittest.TestCase):
    def test_one_day_packet_is_date_removed_and_content_addressed(self) -> None:
        packet, capability, prompt, scorer, model = _packet()

        self.assertEqual(packet["contract"], PACKET_CONTRACT)
        self.assertEqual(prompt["contract"], PROMPT_LOCK_CONTRACT)
        self.assertEqual(capability["contract"], CAPABILITY_MANIFEST_CONTRACT)
        self.assertEqual(scorer["contract"], SCORER_LOCK_CONTRACT)
        self.assertNotIn("decision_cutoff_utc", packet)
        self.assertNotIn("2026-07-19", str(packet))
        self.assertNotIn("f" * 64, str(packet))
        self.assertEqual(
            [row["seconds_before_cutoff"] for row in packet["observations"]],
            [7200, 300],
        )
        self.assertEqual(len(packet["packet_sha256"]), 64)
        self.assertEqual(packet["prompt_lock_sha256"], prompt["prompt_lock_sha256"])
        self.assertEqual(packet["scorer_lock_sha256"], scorer["scorer_lock_sha256"])
        self.assertEqual(
            model["capability_manifest_sha256"],
            capability["capability_manifest_sha256"],
        )

    def test_future_observation_multi_day_container_and_dated_payload_fail_closed(
        self,
    ) -> None:
        capability, prompt, scorer, _model = _locked_artifacts()

        future = _source(
            observations=[
                {
                    "observed_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
                    "kind": "FUTURE_BAR",
                    "payload": {"close": 999.0},
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "after decision cutoff"):
            build_day_packet(
                future,
                prompt_lock=prompt,
                capability_manifest=capability,
                scorer_lock=scorer,
            )

        with self.assertRaisesRegex(ValueError, "single-day source"):
            build_day_packet(  # type: ignore[arg-type]
                [_source(), _source()],
                prompt_lock=prompt,
                capability_manifest=capability,
                scorer_lock=scorer,
            )
        with self.assertRaisesRegex(ValueError, "source shape"):
            build_day_packet(
                {**_source(), "days": [_source(), _source()]},
                prompt_lock=prompt,
                capability_manifest=capability,
                scorer_lock=scorer,
            )

        dated = _source()
        dated["observations"][0]["payload"]["date"] = "2026-07-18"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "date/timestamp"):
            build_day_packet(
                dated,
                prompt_lock=prompt,
                capability_manifest=capability,
                scorer_lock=scorer,
            )

    def test_capability_manifest_declares_fresh_context_and_answer_key_absence(
        self,
    ) -> None:
        capability, _prompt, _scorer, _model = _locked_artifacts()

        self.assertIs(capability["declared_fresh_context"], True)
        self.assertEqual(
            capability["declared_mounted_artifact_roles"], ["INLINE_PACKET_ONLY"]
        )
        self.assertIs(capability["declared_answer_key_physically_absent"], True)
        self.assertIs(capability["declared_filesystem_access"], False)
        self.assertIs(capability["declared_network_access"], False)
        self.assertIs(capability["declared_conversation_history_access"], False)
        self.assertEqual(capability["declared_tools"], [])
        self.assertIs(capability["live_permission"], False)
        self.assertIs(capability["broker_mutation_allowed"], False)

        forged = {**capability, "answer_key_path": "/shared/answer.json"}
        forged.pop("capability_manifest_sha256")
        forged["capability_manifest_sha256"] = canonical_sha256(forged)
        with self.assertRaisesRegex(ValueError, "capability manifest shape"):
            seal_model_manifest(
                model_name="offline-test-model",
                model_version="2026-07-19",
                model_lineage="lineage-a",
                reasoning_effort="high",
                context_id="fresh-context-001",
                capability_manifest=forged,
                locked_at_utc=NOW,
            )

    def test_response_schema_is_fixed_and_receipt_binds_all_prelocked_hashes(
        self,
    ) -> None:
        packet, capability, prompt, _scorer, model = _packet()
        response = seal_response(
            _decision(packet),
            packet=packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            sealed_at_utc=NOW,
        )

        self.assertEqual(response["contract"], RESPONSE_CONTRACT)
        self.assertEqual(set(response["response"]), FIXED_RESPONSE_KEYS)
        for field in (
            "packet_sha256",
            "prompt_lock_sha256",
            "prompt_sha256",
            "model_sha256",
            "capability_manifest_sha256",
            "response_sha256",
            "response_receipt_sha256",
        ):
            self.assertEqual(len(response[field]), 64, field)

        extra = _decision(packet, rationale="free-form repair after outcome")
        with self.assertRaisesRegex(ValueError, "fixed response schema"):
            seal_response(
                extra,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                sealed_at_utc=NOW,
            )

    def test_answer_key_loader_is_not_called_before_valid_response_seal(self) -> None:
        response, packet, capability, prompt, scorer, model = _sealed_response()
        tampered = copy.deepcopy(response)
        tampered["response"]["confidence"] = 0.99
        loader_called = False

        def loader() -> dict[str, object]:
            nonlocal loader_called
            loader_called = True
            return seal_answer_key(
                trial_id=str(packet["trial_id"]),
                packet_sha256=str(packet["packet_sha256"]),
                returns={
                    "FLAT": 0.0,
                    "LONG_HALF": 0.02,
                    "LONG_FULL": 0.04,
                    "SHORT_HALF": -0.02,
                    "SHORT_FULL": -0.04,
                },
                sealed_at_utc=NOW + timedelta(microseconds=1),
            )

        with self.assertRaisesRegex(ValueError, "response receipt seal"):
            score_sealed_response(
                tampered,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=loader,
                opened_at_utc=NOW + timedelta(seconds=1),
            )
        self.assertIs(loader_called, False)

        malformed = copy.deepcopy(response)
        malformed["response"]["post_hoc_rationale"] = "added after outcome"
        malformed["response_sha256"] = canonical_sha256(malformed["response"])
        malformed_body = {
            key: value
            for key, value in malformed.items()
            if key != "response_receipt_sha256"
        }
        malformed["response_receipt_sha256"] = canonical_sha256(malformed_body)
        with self.assertRaisesRegex(ValueError, "fixed response schema"):
            score_sealed_response(
                malformed,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=loader,
                opened_at_utc=NOW + timedelta(seconds=1),
            )
        self.assertIs(loader_called, False)

        with self.assertRaisesRegex(ValueError, "one JSON object"):
            score_sealed_response(
                None,  # type: ignore[arg-type]
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=loader,
                opened_at_utc=NOW + timedelta(seconds=1),
            )
        self.assertIs(loader_called, False)

        score = score_sealed_response(
            response,
            packet=packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            scorer_lock=scorer,
            answer_key_loader=loader,
            opened_at_utc=NOW + timedelta(seconds=1),
        )
        self.assertIs(loader_called, True)
        self.assertEqual(score["contract"], SCORE_CONTRACT)
        self.assertAlmostEqual(score["net_return"], 0.02)

    def test_scorer_must_be_the_exact_prelocked_packet_scorer(self) -> None:
        response, packet, capability, prompt, _scorer, model = _sealed_response()
        replacement_scorer = prelock_scorer(locked_at_utc=NOW - timedelta(seconds=30))
        loader_called = False

        def loader() -> dict[str, object]:
            nonlocal loader_called
            loader_called = True
            raise AssertionError("stale scorer binding must fail before key open")

        with self.assertRaisesRegex(ValueError, "scorer binding"):
            score_sealed_response(
                response,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=replacement_scorer,
                answer_key_loader=loader,
                opened_at_utc=NOW + timedelta(seconds=1),
            )
        self.assertIs(loader_called, False)

    def test_scorer_invalidation_after_response_seal_propagates(self) -> None:
        response, packet, capability, prompt, scorer, model = _sealed_response()
        invalidated_scorer = invalidate_artifact(
            scorer,
            reason="SCORER_POLICY_REVOKED",
            evidence_sha256="c" * 64,
            invalidated_at_utc=NOW + timedelta(milliseconds=1),
        )
        loader_called = False

        def loader() -> dict[str, object]:
            nonlocal loader_called
            loader_called = True
            raise AssertionError("invalidated scorer must block key open")

        child = score_sealed_response(
            response,
            packet=packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            scorer_lock=invalidated_scorer,
            answer_key_loader=loader,
            opened_at_utc=NOW + timedelta(seconds=1),
        )
        self.assertIs(loader_called, False)
        self.assertEqual(child["validity_status"], "INVALIDATED")
        self.assertIsNone(child["net_return"])

    def test_prompt_model_and_scorer_must_be_prelocked_before_response(self) -> None:
        packet, capability, prompt, scorer, _model = _packet()
        future_model = seal_model_manifest(
            model_name="offline-test-model",
            model_version="2026-07-19",
            model_lineage="lineage-a",
            reasoning_effort="high",
            context_id="fresh-context-001",
            capability_manifest=capability,
            locked_at_utc=NOW + timedelta(seconds=1),
        )
        with self.assertRaisesRegex(ValueError, "prelocked before response"):
            seal_response(
                _decision(packet),
                packet=packet,
                prompt_lock=prompt,
                model_manifest=future_model,
                capability_manifest=capability,
                sealed_at_utc=NOW,
            )

        (
            response,
            response_packet,
            response_capability,
            response_prompt,
            _,
            response_model,
        ) = _sealed_response()
        future_scorer = prelock_scorer(locked_at_utc=NOW + timedelta(seconds=1))
        # Preserve the binding to isolate the temporal check.
        response_body = {
            key: value
            for key, value in response.items()
            if key != "response_receipt_sha256"
        }
        response_body["scorer_lock_sha256"] = future_scorer["scorer_lock_sha256"]
        response = {
            **response_body,
            "response_receipt_sha256": canonical_sha256(response_body),
        }
        with self.assertRaisesRegex(ValueError, "prelocked before response"):
            score_sealed_response(
                response,
                packet=response_packet,
                prompt_lock=response_prompt,
                model_manifest=response_model,
                capability_manifest=response_capability,
                scorer_lock=future_scorer,
                answer_key_loader=lambda: {},
                opened_at_utc=NOW + timedelta(seconds=2),
            )

    def test_same_model_lineage_counts_as_one_independent_unit(self) -> None:
        scores: list[dict[str, object]] = []
        for offset in (0, 1):
            response, packet, capability, prompt, scorer, model = _sealed_response(
                lineage="lineage-a"
            )
            if offset:
                # A second valid packet is easiest to create from a shifted source.
                capability, prompt, scorer, model = _locked_artifacts()
                packet = build_day_packet(
                    _source(pair="EUR_USD"),
                    prompt_lock=prompt,
                    capability_manifest=capability,
                    scorer_lock=scorer,
                )
                response = seal_response(
                    _decision(packet, pair="EUR_USD"),
                    packet=packet,
                    prompt_lock=prompt,
                    model_manifest=model,
                    capability_manifest=capability,
                    sealed_at_utc=NOW,
                )
            key = seal_answer_key(
                trial_id=str(packet["trial_id"]),
                packet_sha256=str(packet["packet_sha256"]),
                returns={
                    "FLAT": 0.0,
                    "LONG_HALF": 0.01,
                    "LONG_FULL": 0.02,
                    "SHORT_HALF": -0.01,
                    "SHORT_FULL": -0.02,
                },
                sealed_at_utc=NOW + timedelta(microseconds=1),
            )
            scores.append(
                score_sealed_response(
                    response,
                    packet=packet,
                    prompt_lock=prompt,
                    model_manifest=model,
                    capability_manifest=capability,
                    scorer_lock=scorer,
                    answer_key_loader=lambda key=key: key,
                    opened_at_utc=NOW + timedelta(seconds=1),
                )
            )

        pilot = score_pilot(scores)

        self.assertEqual(pilot["trial_count"], 2)
        self.assertNotIn("effective_independent_n", pilot)
        self.assertEqual(pilot["declared_lineage_trial_counts"], {"lineage-a": 2})
        self.assertEqual(pilot["declared_lineage_cluster_count_diagnostic"], 1)
        self.assertIs(pilot["live_permission"], False)
        self.assertIs(pilot["proof_eligible"], False)

    def test_invalidation_propagates_without_opening_key(self) -> None:
        response, packet, capability, prompt, scorer, model = _sealed_response()
        invalidated_response = invalidate_artifact(
            response,
            reason="LOOKAHEAD_PACKET_CONTAMINATION",
            evidence_sha256="e" * 64,
            invalidated_at_utc=NOW + timedelta(milliseconds=1),
        )
        loader_called = False

        def loader() -> dict[str, object]:
            nonlocal loader_called
            loader_called = True
            raise AssertionError("invalidated child must not open answer key")

        child = score_sealed_response(
            invalidated_response,
            packet=packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            scorer_lock=scorer,
            answer_key_loader=loader,
            opened_at_utc=NOW + timedelta(seconds=1),
        )
        self.assertIs(loader_called, False)
        self.assertEqual(child["validity_status"], "INVALIDATED")
        self.assertIsNone(child["net_return"])
        pilot = score_pilot([child])
        self.assertEqual(pilot["validity_status"], "INVALIDATED")
        self.assertIsNone(pilot["total_log_growth"])
        self.assertNotIn("positive_evidence", pilot)

    def test_packet_and_positive_score_invalidation_propagate_transitively(
        self,
    ) -> None:
        packet, capability, prompt, scorer, model = _packet()
        invalidated_packet = invalidate_artifact(
            packet,
            reason="PACKET_CONTAMINATION_DISCOVERED",
            evidence_sha256="a" * 64,
            invalidated_at_utc=NOW + timedelta(seconds=1),
        )
        response = seal_response(
            _decision(packet),
            packet=invalidated_packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            sealed_at_utc=NOW + timedelta(seconds=2),
        )
        self.assertEqual(response["validity_status"], "INVALIDATED")

        (
            valid_response,
            valid_packet,
            valid_capability,
            valid_prompt,
            valid_scorer,
            valid_model,
        ) = _sealed_response()
        answer_key = seal_answer_key(
            trial_id=str(valid_packet["trial_id"]),
            packet_sha256=str(valid_packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.02,
                "LONG_FULL": 0.04,
                "SHORT_HALF": -0.02,
                "SHORT_FULL": -0.04,
            },
            sealed_at_utc=NOW + timedelta(microseconds=1),
        )
        positive_score = score_sealed_response(
            valid_response,
            packet=valid_packet,
            prompt_lock=valid_prompt,
            model_manifest=valid_model,
            capability_manifest=valid_capability,
            scorer_lock=valid_scorer,
            answer_key_loader=lambda: answer_key,
            opened_at_utc=NOW + timedelta(seconds=1),
        )
        invalidated_score = invalidate_artifact(
            positive_score,
            reason="PARENT_RESPONSE_INVALIDATED",
            evidence_sha256="b" * 64,
            invalidated_at_utc=NOW + timedelta(seconds=2),
        )
        pilot = score_pilot([invalidated_score])
        self.assertEqual(pilot["validity_status"], "INVALIDATED")
        self.assertIsNone(pilot["total_log_growth"])

    def test_stale_positive_artifact_fails_closed(self) -> None:
        stale = {
            "contract": "QR_AI_DISCRETION_DAYREAD_PILOT_V1",
            "verdict": "FIRST_POSITIVE_MACHINE_EVIDENCE_FOR_LAYER_C",
            "results": {"commit_win_rate": "27/31"},
        }
        with self.assertRaisesRegex(ValueError, "stale positive artifact"):
            assert_no_stale_positive_artifacts([stale])

        response, packet, capability, prompt, scorer, model = _sealed_response()
        answer_key = seal_answer_key(
            trial_id=str(packet["trial_id"]),
            packet_sha256=str(packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.02,
                "LONG_FULL": 0.04,
                "SHORT_HALF": -0.02,
                "SHORT_FULL": -0.04,
            },
            sealed_at_utc=NOW + timedelta(microseconds=1),
        )
        positive = score_sealed_response(
            response,
            packet=packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            scorer_lock=scorer,
            answer_key_loader=lambda: answer_key,
            opened_at_utc=NOW + timedelta(seconds=1),
        )
        self.assertNotIn("positive_evidence", positive)
        self.assertEqual(positive["diagnostic_status"], "DIAGNOSTIC_ONLY")

    def test_positive_numeric_metric_is_stale_even_without_verdict(self) -> None:
        with self.assertRaisesRegex(ValueError, "validity is missing or not VALID"):
            assert_no_stale_positive_artifacts(
                [
                    {
                        "contract": SCORE_CONTRACT,
                        "validity_status": "INVALIDATED",
                        "diagnostic_return_sign": "POSITIVE",
                        "net_return": 0.01,
                    }
                ]
            )

    def test_answer_key_sealed_before_response_is_rejected(self) -> None:
        response, packet, capability, prompt, scorer, model = _sealed_response()
        early_key = seal_answer_key(
            trial_id=str(packet["trial_id"]),
            packet_sha256=str(packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.01,
                "LONG_FULL": 0.02,
                "SHORT_HALF": -0.01,
                "SHORT_FULL": -0.02,
            },
            sealed_at_utc=NOW - timedelta(days=1),
        )
        with self.assertRaisesRegex(ValueError, "sealed after response"):
            score_sealed_response(
                response,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=lambda: early_key,
                opened_at_utc=NOW + timedelta(seconds=1),
            )

    def test_resealed_diagnostic_parent_cannot_grant_live_permission(self) -> None:
        capability, prompt, scorer, _model = _locked_artifacts()
        forged_body = {
            key: value for key, value in prompt.items() if key != "prompt_lock_sha256"
        }
        forged_body["live_permission"] = True
        forged = {
            **forged_body,
            "prompt_lock_sha256": canonical_sha256(forged_body),
        }
        with self.assertRaisesRegex(ValueError, "grants live permission"):
            build_day_packet(
                _source(),
                prompt_lock=forged,
                capability_manifest=capability,
                scorer_lock=scorer,
            )

    def test_packet_builder_cli_emits_only_offline_prelocked_bundle(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.json"
            output_dir = root / "bundle"
            source_path.write_text(json.dumps(_source()), encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repo / "src")
            command = [
                sys.executable,
                str(repo / "scripts/build-dojo-ai-packet.py"),
                "--source",
                str(source_path),
                "--variant-id",
                "A_FABLE_MINIMAL",
                "--phase-id",
                "phase_1_diagnostic",
                "--blind-day-rank",
                "1",
                "--blind-day-id",
                "b" * 64,
                "--context-id",
                "fresh-cli-context",
                "--model-name",
                "offline-test-model",
                "--model-version",
                "v1",
                "--model-lineage",
                "lineage-cli",
                "--reasoning-effort",
                "high",
                "--locked-at-utc",
                (NOW - timedelta(minutes=1)).isoformat(),
                "--output-dir",
                str(output_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            result = json.loads(completed.stdout)
            request_path = Path(result["artifacts"]["request"])
            packet_path = Path(result["artifacts"]["packet"])
            request = json.loads(request_path.read_text())
            packet = json.loads(packet_path.read_text())
            builder_sha_fields = {
                "capability": "capability_manifest_sha256",
                "prompt": "prompt_lock_sha256",
                "scorer": "scorer_lock_sha256",
                "model": "model_sha256",
                "packet": "packet_sha256",
                "request": "request_receipt_sha256",
                "cell": "cell_id",
            }
            for role, artifact_path_text in result["artifacts"].items():
                artifact_path = Path(artifact_path_text)
                artifact = json.loads(artifact_path.read_text())
                self.assertIn(
                    str(artifact[builder_sha_fields[role]]), artifact_path.name
                )
                if role != "cell":
                    self.assertEqual(artifact["evidence_tier"], DIAGNOSTIC_TIER)
                self.assertEqual(artifact_path.stat().st_mode & 0o777, 0o600)
            repeated = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result["status"], "DOJO_AI_PACKET_LOCKED")
        self.assertIs(request["declared_answer_key_must_be_physically_absent"], True)
        self.assertIs(request["model_api_invoked"], False)
        self.assertIs(request["live_permission"], False)
        self.assertEqual(request["evidence_tier"], DIAGNOSTIC_TIER)
        self.assertIn(str(request["request_receipt_sha256"]), request_path.name)
        self.assertIn(str(packet["packet_sha256"]), packet_path.name)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("must be new", repeated.stderr)
        self.assertNotIn("2026-07-19", json.dumps(packet))

    def test_score_cli_reads_key_only_for_a_sealed_response(self) -> None:
        response, packet, capability, prompt, scorer, model = (
            _sealed_response_with_parents()
        )
        answer_key = seal_answer_key(
            trial_id=str(packet["trial_id"]),
            packet_sha256=str(packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.01,
                "LONG_FULL": 0.02,
                "SHORT_HALF": -0.01,
                "SHORT_FULL": -0.02,
            },
            sealed_at_utc=NOW + timedelta(microseconds=1),
        )
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            response_path = root / "response.json"
            key_path = root / "answer.json"
            scorer_path = root / "scorer.json"
            packet_path = root / "packet.json"
            prompt_path = root / "prompt.json"
            model_path = root / "model.json"
            capability_path = root / "capability.json"
            output_dir = root / "scores"
            for path, value in (
                (response_path, response),
                (key_path, answer_key),
                (scorer_path, scorer),
                (packet_path, packet),
                (prompt_path, prompt),
                (model_path, model),
                (capability_path, capability),
            ):
                path.write_text(json.dumps(value), encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repo / "src")
            command = [
                sys.executable,
                str(repo / "scripts/score-dojo-ai-pilot.py"),
                "--legacy-per-trial-diagnostic",
                "--response",
                str(response_path),
                "--answer-key",
                str(key_path),
                "--packet",
                str(packet_path),
                "--prompt-lock",
                str(prompt_path),
                "--model-manifest",
                str(model_path),
                "--capability-manifest",
                str(capability_path),
                "--scorer-lock",
                str(scorer_path),
                "--opened-at-utc",
                (NOW + timedelta(seconds=1)).isoformat(),
                "--output-dir",
                str(output_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            no_acknowledgement = list(command)
            no_acknowledgement.remove("--legacy-per-trial-diagnostic")
            refused = subprocess.run(
                no_acknowledgement,
                cwd=repo,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(completed.stdout)
            bundle_path = Path(result["artifacts"]["bundle"])
            bundle = json.loads(bundle_path.read_text())
            for artifact_path_text in result["artifacts"].values():
                artifact_path = Path(artifact_path_text)
                artifact = json.loads(artifact_path.read_text())
                self.assertEqual(artifact["evidence_tier"], DIAGNOSTIC_TIER)
                self.assertEqual(artifact_path.stat().st_mode & 0o777, 0o600)
            repeated = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result["status"], "VALID")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("registered 90-cell denominator", refused.stderr)
        self.assertIs(result["legacy_per_trial_diagnostic"], True)
        self.assertIs(result["registered_experiment_evidence"], False)
        self.assertIs(result["prompt_selection_allowed"], False)
        self.assertNotIn("effective_independent_n", result)
        self.assertEqual(result["declared_lineage_cluster_count_diagnostic"], 1)
        self.assertAlmostEqual(bundle["scores"][0]["net_return"], 0.01)
        self.assertIs(
            bundle["scores"][0]["parent_bindings_revalidated_in_process"], True
        )
        self.assertIs(bundle["live_permission"], False)
        self.assertIs(bundle["model_api_invoked"], False)
        self.assertIs(bundle["registered_experiment_evidence"], False)
        self.assertIs(bundle["registered_phase_denominator_enforced"], False)
        self.assertEqual(bundle["evidence_tier"], DIAGNOSTIC_TIER)
        self.assertIn(str(bundle["bundle_sha256"]), bundle_path.name)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("must be new", repeated.stderr)

    def test_unattested_artifacts_never_claim_edge_or_independent_n(self) -> None:
        response, packet, capability, prompt, scorer, model = _sealed_response()
        answer_key = seal_answer_key(
            trial_id=str(packet["trial_id"]),
            packet_sha256=str(packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.50,
                "LONG_FULL": 0.90,
                "SHORT_HALF": -0.25,
                "SHORT_FULL": -0.50,
            },
            sealed_at_utc=NOW + timedelta(microseconds=1),
        )
        score = score_sealed_response(
            response,
            packet=packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            scorer_lock=scorer,
            answer_key_loader=lambda: answer_key,
            opened_at_utc=NOW + timedelta(seconds=1),
        )
        pilot = score_pilot([score])
        registry = seal_validity_registry(
            {str(score["score_receipt_sha256"]): "VALID"},
            updated_at_utc=NOW + timedelta(seconds=2),
        )

        for artifact in (
            capability,
            prompt,
            scorer,
            model,
            packet,
            response,
            answer_key,
            score,
            pilot,
            registry,
        ):
            self.assertEqual(artifact["evidence_tier"], DIAGNOSTIC_TIER)
            self.assertIs(artifact["external_attestations_verified"], False)
            self.assertNotIn("positive_evidence", artifact)
            self.assertNotIn("effective_independent_n", artifact)
            self.assertNotIn("EDGE_PROVEN", json.dumps(artifact))

    def test_public_sha_response_reseal_is_revalidated_against_packet(self) -> None:
        response, packet, capability, prompt, scorer, model = _sealed_response()
        forged = copy.deepcopy(response)
        forged["response"]["evidence_refs"] = ["obs-999"]
        forged["response_sha256"] = canonical_sha256(forged["response"])
        forged_body = {
            key: value
            for key, value in forged.items()
            if key != "response_receipt_sha256"
        }
        forged["response_receipt_sha256"] = canonical_sha256(forged_body)
        loader_called = False

        def loader() -> dict[str, object]:
            nonlocal loader_called
            loader_called = True
            raise AssertionError("forged response must fail before key open")

        with self.assertRaisesRegex(ValueError, "outside its packet"):
            score_sealed_response(
                forged,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=loader,
                opened_at_utc=NOW + timedelta(seconds=1),
            )
        self.assertIs(loader_called, False)

        lineage_forged = {
            **response,
            "model_lineage_attestation_status": "PROVIDER_ATTESTED",
        }
        lineage_body = {
            key: value
            for key, value in lineage_forged.items()
            if key != "response_receipt_sha256"
        }
        lineage_forged["response_receipt_sha256"] = canonical_sha256(lineage_body)
        with self.assertRaisesRegex(ValueError, "declared only"):
            score_sealed_response(
                lineage_forged,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=loader,
                opened_at_utc=NOW + timedelta(seconds=1),
            )
        self.assertIs(loader_called, False)

    def test_public_sha_score_reseal_cannot_forge_score_math(self) -> None:
        response, packet, capability, prompt, scorer, model = _sealed_response()
        answer_key = seal_answer_key(
            trial_id=str(packet["trial_id"]),
            packet_sha256=str(packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.01,
                "LONG_FULL": 0.02,
                "SHORT_HALF": -0.01,
                "SHORT_FULL": -0.02,
            },
            sealed_at_utc=NOW + timedelta(microseconds=1),
        )
        score = score_sealed_response(
            response,
            packet=packet,
            prompt_lock=prompt,
            model_manifest=model,
            capability_manifest=capability,
            scorer_lock=scorer,
            answer_key_loader=lambda: answer_key,
            opened_at_utc=NOW + timedelta(seconds=1),
        )
        forged = {**score, "net_return": 0.99}
        forged_body = {
            key: value for key, value in forged.items() if key != "score_receipt_sha256"
        }
        forged["score_receipt_sha256"] = canonical_sha256(forged_body)
        with self.assertRaisesRegex(ValueError, "score math"):
            score_pilot([forged])

        promotion_forged = {**score, "positive_evidence": True}
        promotion_body = {
            key: value
            for key, value in promotion_forged.items()
            if key != "score_receipt_sha256"
        }
        promotion_forged["score_receipt_sha256"] = canonical_sha256(promotion_body)
        with self.assertRaisesRegex(ValueError, "proof or promotion"):
            score_pilot([promotion_forged])

        verdict_forged = {**score, "verdict": "PASS"}
        verdict_body = {
            key: value
            for key, value in verdict_forged.items()
            if key != "score_receipt_sha256"
        }
        verdict_forged["score_receipt_sha256"] = canonical_sha256(verdict_body)
        with self.assertRaisesRegex(ValueError, "proof or promotion"):
            score_pilot([verdict_forged])

        lineage_forged = {
            **score,
            "model_lineage_attestation_status": "PROVIDER_ATTESTED",
        }
        lineage_body = {
            key: value
            for key, value in lineage_forged.items()
            if key != "score_receipt_sha256"
        }
        lineage_forged["score_receipt_sha256"] = canonical_sha256(lineage_body)
        with self.assertRaisesRegex(ValueError, "declared only"):
            score_pilot([lineage_forged])

    def test_public_sha_cannot_upgrade_declared_lineage_or_answer_provenance(
        self,
    ) -> None:
        response, packet, capability, prompt, scorer, model = (
            _sealed_response_with_parents()
        )
        forged_model = {
            **model,
            "model_lineage_attestation_status": "PROVIDER_ATTESTED",
        }
        forged_model_body = {
            key: value for key, value in forged_model.items() if key != "model_sha256"
        }
        forged_model["model_sha256"] = canonical_sha256(forged_model_body)
        with self.assertRaisesRegex(ValueError, "declared only"):
            seal_response(
                _decision(packet),
                packet=packet,
                prompt_lock=prompt,
                model_manifest=forged_model,
                capability_manifest=capability,
                sealed_at_utc=NOW,
            )

        answer_key = seal_answer_key(
            trial_id=str(packet["trial_id"]),
            packet_sha256=str(packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.01,
                "LONG_FULL": 0.02,
                "SHORT_HALF": -0.01,
                "SHORT_FULL": -0.02,
            },
            sealed_at_utc=NOW + timedelta(microseconds=1),
        )
        forged_key = {
            **answer_key,
            "answer_key_provenance_status": "MARKET_PROVIDER_ATTESTED",
            "market_data_attestation_sha256": "a" * 64,
        }
        forged_key_body = {
            key: value
            for key, value in forged_key.items()
            if key != "answer_key_sha256"
        }
        forged_key["answer_key_sha256"] = canonical_sha256(forged_key_body)
        with self.assertRaisesRegex(ValueError, "overstates trusted market provenance"):
            score_sealed_response(
                response,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=lambda: forged_key,
                opened_at_utc=NOW + timedelta(seconds=1),
            )

    def test_numeric_date_epoch_and_event_encodings_are_rejected(self) -> None:
        capability, prompt, scorer, _model = _locked_artifacts()
        for encoded in (
            20260719,
            1_753_000_000,
            1_753_000_000_000,
            "20260719070000",
            "event-20260719",
            "epoch=1753000000",
        ):
            source = _source()
            source["observations"][0]["payload"]["event_id"] = encoded  # type: ignore[index]
            with self.subTest(encoded=encoded):
                with self.assertRaisesRegex(ValueError, "date/epoch encoding"):
                    build_day_packet(
                        source,
                        prompt_lock=prompt,
                        capability_manifest=capability,
                        scorer_lock=scorer,
                    )

        response, packet, capability, prompt, scorer, model = _sealed_response()
        forged_packet = copy.deepcopy(packet)
        forged_packet["observations"][0]["payload"]["event_id"] = "event-20260719"
        forged_packet_body = {
            key: value for key, value in forged_packet.items() if key != "packet_sha256"
        }
        forged_packet["packet_sha256"] = canonical_sha256(forged_packet_body)
        loader_called = False

        def loader() -> dict[str, object]:
            nonlocal loader_called
            loader_called = True
            raise AssertionError("date-bearing packet must fail before key open")

        with self.assertRaisesRegex(ValueError, "date/epoch encoding"):
            score_sealed_response(
                response,
                packet=forged_packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer,
                answer_key_loader=loader,
                opened_at_utc=NOW + timedelta(seconds=1),
            )
        self.assertIs(loader_called, False)

    def test_answer_key_and_registry_are_honestly_unattested(self) -> None:
        response, packet, _capability, _prompt, scorer, _model = _sealed_response()
        answer_key = seal_answer_key(
            trial_id=str(packet["trial_id"]),
            packet_sha256=str(packet["packet_sha256"]),
            returns={
                "FLAT": 0.0,
                "LONG_HALF": 0.01,
                "LONG_FULL": 0.02,
                "SHORT_HALF": -0.01,
                "SHORT_FULL": -0.02,
            },
            sealed_at_utc=NOW + timedelta(microseconds=1),
        )
        registry = seal_validity_registry(
            {str(response["response_receipt_sha256"]): "VALID"},
            updated_at_utc=NOW,
        )
        self.assertEqual(
            answer_key["answer_key_provenance_status"], "SELF_ATTESTED_UNVERIFIED"
        )
        self.assertEqual(registry["monotonicity_status"], "NOT_EXTERNALLY_MONOTONIC")
        self.assertEqual(
            scorer["policy"]["missing_or_schema_invalid_response"],
            "REJECT_BEFORE_KEY_OPEN",
        )

    def test_pilot_return_is_compounded_not_summed(self) -> None:
        scores: list[dict[str, object]] = []
        for pair in ("USD_JPY", "EUR_USD"):
            capability, prompt, scorer, model = _locked_artifacts()
            packet = build_day_packet(
                _source(pair=pair),
                prompt_lock=prompt,
                capability_manifest=capability,
                scorer_lock=scorer,
            )
            response = seal_response(
                _decision(packet, pair=pair),
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                sealed_at_utc=NOW,
            )
            key = seal_answer_key(
                trial_id=str(packet["trial_id"]),
                packet_sha256=str(packet["packet_sha256"]),
                returns={
                    "FLAT": 0.0,
                    "LONG_HALF": 0.10,
                    "LONG_FULL": 0.20,
                    "SHORT_HALF": -0.10,
                    "SHORT_FULL": -0.20,
                },
                sealed_at_utc=NOW + timedelta(microseconds=1),
            )
            scores.append(
                score_sealed_response(
                    response,
                    packet=packet,
                    prompt_lock=prompt,
                    model_manifest=model,
                    capability_manifest=capability,
                    scorer_lock=scorer,
                    answer_key_loader=lambda key=key: key,
                    opened_at_utc=NOW + timedelta(seconds=1),
                )
            )
        pilot = score_pilot(scores)
        self.assertAlmostEqual(pilot["compounded_net_return"], 0.21)

    def test_packet_label_admits_exact_price_reidentification_risk(self) -> None:
        packet, _capability, _prompt, _scorer, _model = _packet()
        self.assertIs(packet["calendar_date_removed"], True)
        self.assertEqual(packet["anonymity_status"], "PSEUDONYMIZED_NOT_ANONYMOUS")
        self.assertEqual(
            packet["reidentification_risk"],
            "EXACT_PRICE_PATH_MAY_REIDENTIFY_MARKET_DATE",
        )


if __name__ == "__main__":
    unittest.main()
